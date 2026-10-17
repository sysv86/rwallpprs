//! Интеграция с протоколом Wayland `zwlr_layer_shell_v1` для композитора **Niri**.

use anyhow::Context as _;
use rwallpprs_ipc::ShmBufferHeader;
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{slot::SlotPool, Shm, ShmHandler},
};
use std::{os::fd::BorrowedFd, os::fd::RawFd, ptr::NonNull};
use rustix::fs::fstat;
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_buffer, wl_output, wl_shm, wl_shm_pool, wl_surface},
    Connection, Dispatch, QueueHandle,
};
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};

/// Zero-copy direct present: `wl_shm`-пул создаётся поверх того же `memfd`,
/// куда рендерит движок (Rust-бинарник через SHM-протокол, видео — через
/// собственное memfd-кольцо демона). Кадр уходит композитору **без единого
/// memcpy**; основной цикл fallback'ается на копирующий `present_frame`,
/// если direct-режим недоступен.
pub struct DirectShmPresent {
    /// Удерживаем живым, пока буферы из него активны на поверхности
    #[allow(dead_code)]
    pool: wl_shm_pool::WlShmPool,
    /// `(wl_buffer, released)`: `true` => сервер отдал буфер, слот можно заполнять
    buffers: Vec<(wl_buffer::WlBuffer, bool)>,
    width: u32,
    height: u32,
    /// fd и размеры memfd-источника, поверх которого создан пул. Нужны для проверки
    /// согласованности в главном цикле демона: после смены движка (SetVideo /
    /// SetRustBin / SetEmbeddedBin) запоротый презентер иначе продолжил бы
    /// показывать композитору буферы чужого memfd.
    source_fd: RawFd,
    /// Identity файла-источника (dev/ino из fstat): номера fd переиспользуются ядром
    /// после закрытия, поэтому по одному номеру fd чужой memfd не отличить от нового.
    source_dev: u64,
    source_ino: u64,
    /// Второй mmap демона того же memfd (указывает ровно на слоты движка).
    /// Нужен для записи кадров (видео) прямо в память композитора без копий.
    shm_ptr: NonNull<u8>,
    slot_size_bytes: usize,
}

impl DirectShmPresent {
    /// Импортирует memfd движка (`direct_fd`) как `wl_shm`-пул и создаёт по одному
    /// `wl_buffer` поверх слот-офсетов `ShmBufferHeader` (0 и 1), шаг `width*4`,
    /// формат `XRGB8888`. `udata=slot` для связки release-событий со слотами.
    ///
    /// `write_header=true` — для memfd-кольца демона под видео: заголовок
    /// появляется только здесь. Для memfd Rust-раннера передавай `false`:
    /// заголовок уже записан `spawn_internal`.
    pub fn init(
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<WaylandLayerContext>,
        direct_fd: RawFd,
        width: u32,
        height: u32,
        write_header: bool,
    ) -> anyhow::Result<Self> {
        let header = ShmBufferHeader::new(width, height);
        let total_size = header.total_shm_size();

        let borrowed_fd = unsafe { BorrowedFd::borrow_raw(direct_fd) };

        // mmap первым: при неудаче здесь ещё нет ни wl_shm_pool, ни wl_buffer —
        // чистить нечего, err-путь остаётся прозрачным
        let raw_ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                total_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                borrowed_fd,
                0,
            )
            .context("Ошибка rustix::mm::mmap демона для direct-present")?
        };
        let shm_ptr = match NonNull::new(raw_ptr.cast::<u8>()) {
            Some(p) => p,
            None => {
                unsafe { let _ = munmap(raw_ptr.cast(), total_size); }
                anyhow::bail!("mmap вернул нулевой указатель");
            }
        };

        // Identity файла-источника: dev/ino понадобится главному циклу, чтобы
        // отличить этот memfd от нового под тем же (переиспользованным) номером fd
        let stat = match fstat(borrowed_fd) {
            Ok(stat) => stat,
            Err(e) => {
                unsafe { let _ = munmap(shm_ptr.as_ptr().cast(), total_size); }
                return Err(e).context("Ошибка rustix::fs::fstat источника direct-present");
            }
        };

        // Демон сам маппит тот же memfd, чтобы записывать кадры прямо в слоты без промежуточных буферов
        let pool = shm.create_pool(borrowed_fd, total_size as i32, qh, ());

        let buffers = (0..2usize)
            .map(|i| {
                let buffer = pool.create_buffer(
                    header.slot_offset(i as u8) as i32,
                    width as i32,
                    height as i32,
                    header.stride as i32,
                    wl_shm::Format::Xrgb8888,
                    qh,
                    i,
                );
                // Сервер ещё ничего не показывал => оба слота формально свободны
                (buffer, true)
            })
            .collect::<Vec<_>>();

        // Заголовок переписывается только там, где его ещё нет: для memfd демона
        // под видео он появляется именно здесь; memfd Rust-раннера уже размечен.
        if write_header {
            unsafe { std::ptr::write(shm_ptr.as_ptr().cast::<ShmBufferHeader>(), header) };
        }

        Ok(Self {
            pool,
            buffers,
            width,
            height,
            source_fd: direct_fd,
            source_dev: stat.st_dev,
            source_ino: stat.st_ino,
            shm_ptr,
            slot_size_bytes: header.slot_size_bytes as usize,
        })
    }

    /// Прямой `&mut`-срез слота memfd (0/1) для записи кадра движком — ноль копий.
    /// `None`, если слот всё ещё в композиторе (перезаписывать нельзя).
    unsafe fn slot_slice_mut(&mut self, slot: usize) -> Option<&mut [u8]> {
        let slot = slot & 1;
        if !self.buffers[slot].1 {
            return None;
        }
        let ptr = self
            .shm_ptr
            .as_ptr()
            .add(ShmBufferHeader::HEADER_SIZE + slot * self.slot_size_bytes);
        Some(std::slice::from_raw_parts_mut(ptr, self.slot_size_bytes))
    }
}

impl Drop for DirectShmPresent {
    fn drop(&mut self) {
        // Явно отправляем запросы wl_buffer.destroy для обоих буферов и
        // wl_shm_pool.destroy, чтобы композитор отпустил ссылку на память memfd;
        // у typed-proxy wayland-client 0.31 destroy — это сгенерированный запрос.
        for (buffer, _) in &self.buffers {
            buffer.destroy();
        }
        self.pool.destroy();

        // Отдаём mmap демона: композитор держит собственное отображение по своему fd,
        // поэтому здесь закрывается только виртуальная память нашего процесса.
        let total_size = ShmBufferHeader::HEADER_SIZE + self.slot_size_bytes * 2;
        unsafe {
            let _ = munmap(self.shm_ptr.as_ptr().cast(), total_size);
        }
    }
}

/// Проверка identity открытого файла: свёрка dev/ino через fstat. Номер fd
/// переиспользуется после закрытия, поэтому один только номер ничего не гарантирует.
fn file_identity_matches(fd: RawFd, source_dev: u64, source_ino: u64) -> bool {
    let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
    match fstat(borrowed) {
        Ok(stat) => (stat.st_dev, stat.st_ino) == (source_dev, source_ino),
        Err(_) => false,
    }
}

pub struct WaylandLayerContext {
    pub registry_state: RegistryState,
    pub output_state: OutputState,
    pub compositor_state: CompositorState,
    pub shm_state: Shm,
    pub layer_shell: LayerShell,
    pub pool: Option<SlotPool>,
    /// Zero-copy direct-презентер поверх memfd движка; `None` => копирующий `present_frame`
    pub direct: Option<DirectShmPresent>,
    pub layer_surface: Option<LayerSurface>,
    pub width: u32,
    pub height: u32,
    pub configured: bool,
    pub resolution_changed: bool,
    pub surface_hidden: bool,
}

impl WaylandLayerContext {
    pub fn connect_and_init(
    ) -> anyhow::Result<(Connection, wayland_client::EventQueue<Self>, Self)> {
        let conn = Connection::connect_to_env()?;
        let (globals, mut event_queue) = registry_queue_init(&conn)?;
        let qh = event_queue.handle();

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);
        let compositor_state = CompositorState::bind(&globals, &qh)?;
        let shm_state = Shm::bind(&globals, &qh)?;
        let layer_shell = LayerShell::bind(&globals, &qh)?;

        let surface = compositor_state.create_surface(&qh);
        let layer_surface = layer_shell.create_layer_surface(
            &qh,
            surface,
            Layer::Background,
            Some("rwallpprs-niri"),
            None,
        );

        layer_surface.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer_surface.set_exclusive_zone(-1);
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer_surface.commit();

        let mut ctx = Self {
            registry_state,
            output_state,
            compositor_state,
            shm_state,
            layer_shell,
            pool: None,
            direct: None,
            layer_surface: Some(layer_surface),
            width: 1920,
            height: 1080,
            configured: false,
            resolution_changed: false,
            surface_hidden: false,
        };

        event_queue.roundtrip(&mut ctx)?;
        Ok((conn, event_queue, ctx))
    }

    pub fn present_frame(&mut self, pixels_bgra: &[u8]) -> anyhow::Result<()> {
        if !self.configured {
            anyhow::bail!("present_frame: поверхность еще не сконфигурирована compositor'ом");
        }
        let width = self.width;
        let height = self.height;
        let stride = width as i32 * 4;
        let expected_bytes = (width as usize) * (height as usize) * 4;

        if pixels_bgra.len() < expected_bytes {
            anyhow::bail!(
                "present_frame: кадр {} байт < ожидаемых {} ({}x{})",
                pixels_bgra.len(),
                expected_bytes,
                width,
                height
            );
        }

        if self.pool.is_none() {
            self.pool = Some(SlotPool::new(expected_bytes * 2, &self.shm_state)?);
        }

        let pool = self.pool.as_mut().unwrap();
        let (buffer, canvas) = pool.create_buffer(
            width as i32,
            height as i32,
            stride,
            wl_shm::Format::Xrgb8888,
        )?;

        canvas[..expected_bytes].copy_from_slice(&pixels_bgra[..expected_bytes]);

        if let Some(layer_surface) = &self.layer_surface {
            let wl_surf = layer_surface.wl_surface();
            buffer.attach_to(wl_surf)?;
            wl_surf.damage_buffer(0, 0, width as i32, height as i32);
            wl_surf.commit();
            self.surface_hidden = false;
        }
        Ok(())
    }

    /// Поднимает zero-copy direct present: memfd `direct_fd` импортируется в `wl_shm`
    /// как пул, композитор отображает ровно ту память демона, куда рендерит движок.
    /// Возврат `Err` — сигнал основному циклу откатиться на копирующий `present_frame`.
    ///
    /// `write_header=true` — только для видео-кольца демона (заголовок появляется
    /// тут); для memfd Rust-раннера передавай `false` (заголовок уже записан).
    pub fn init_direct_shm(
        &mut self,
        qh: &QueueHandle<Self>,
        direct_fd: RawFd,
        width: u32,
        height: u32,
        write_header: bool,
    ) -> anyhow::Result<()> {
        if !self.configured {
            anyhow::bail!("init_direct_shm: поверхность не сконфигурирована compositor'ом");
        }
        if self.direct.is_some() {
            anyhow::bail!("init_direct_shm: direct-режим уже активен");
        }
        let shm = self.shm_state.wl_shm().clone();
        self.direct = Some(DirectShmPresent::init(
            &shm, qh, direct_fd, width, height, write_header,
        )?);
        Ok(())
    }

    /// Гарантия согласованности direct-презентера с текущим движком: если
    /// презентер указывает на чужой/недоступный источник (`None`) или fd не
    /// совпадает с текущим (`Some`) — сбрасываем его. Для видео вызывается с fd
    /// текущего `VideoDirectRing`, для Rust-бинарника — с `runner.shm_raw_fd()`.
    pub fn reset_direct_if_stale(&mut self, source: Option<(RawFd, u32, u32)>) {
        let stale = match (&self.direct, source) {
            (Some(d), Some((fd, w, h))) => {
                (d.source_fd, d.width, d.height) != (fd, w, h) || !file_identity_matches(fd, d.source_dev, d.source_ino)
            }
            (Some(_), None) => true,
            (None, _) => false,
        };
        if stale {
            self.direct = None;
        }
    }

    /// Свободен ли слот direct-презентера для следующего кадра (композитор уже
    /// отдал release). `true`, когда direct отсутствует: flow-control не следует
    /// гейтить путь, где его ещё/вновь нет — там работает копирующий fallback.
    pub fn direct_slot_free(&self, slot: u8) -> bool {
        match &self.direct {
            Some(direct) => direct
                .buffers
                .get((slot & 1) as usize)
                .map_or(true, |(_, released)| *released),
            None => true,
        }
    }

    /// Zero-copy показ слота: attach прямого `wl_buffer` поверх memfd + полный damage.
    /// `Ok(true)` — кадр ушёл на экран; `Ok(false)` — direct недоступен или слот занят
    /// (сервер ещё не отдал буфер): основной цикл fallback'ается на `present_frame`.
    pub fn present_shm_slot(&mut self, slot: usize) -> anyhow::Result<bool> {
        if !self.configured {
            anyhow::bail!("present_shm_slot: поверхность еще не сконфигурирована compositor'ом");
        }
        let slot = slot & 1;

        // Раздельные заимствования полей: layer_surface (иммутабельный) и direct (мутабельный)
        match (&self.layer_surface, self.direct.as_mut()) {
            (Some(layer_surface), Some(direct)) => {
                if !direct.buffers[slot].1 {
                    // Слот ещё в композиторе (release не пришёл) — копирующий путь
                    return Ok(false);
                }
                let wl_surf = layer_surface.wl_surface();
                // attach прямой памяти memfd => пиксели идут композитору без копий
                wl_surf.attach(Some(&direct.buffers[slot].0), 0, 0);
                wl_surf.damage_buffer(0, 0, direct.width as i32, direct.height as i32);
                wl_surf.commit();
                self.surface_hidden = false;
                direct.buffers[slot].1 = false;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Поручает `&mut`-срез слота memfd (0/1) для прямой записи кадра движком
    /// (например, `video.decode_next_frame_into`). `None` => копирующий путь.
    pub fn direct_slot_slice(&mut self, slot: usize) -> Option<&mut [u8]> {
        let direct = self.direct.as_mut()?;
        unsafe { direct.slot_slice_mut(slot) }
    }

    /// Пометка "обои скрыты от рисования" без отсоединения буфера поверхности:
    /// реальное отцепление wl-буфера на layer-поверхности порождает у Niri
    /// протокольную ошибку и убивает слой (см. место вызова в main.rs).
    pub fn mark_hidden_for_powersave(&mut self) {
        self.surface_hidden = true;
    }
}

impl CompositorHandler for WaylandLayerContext {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for WaylandLayerContext {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for WaylandLayerContext {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {}

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let new_w = configure.new_size.0.max(640);
        let new_h = configure.new_size.1.max(360);
        if new_w != self.width || new_h != self.height {
            self.width = new_w;
            self.height = new_h;
            self.pool = None;
            // Direct-презентер рассчитан на прежний layout memfd — уходим на копирующий
            // fallback; пересоздание произойдёт лениво на первом кадре движка
            self.direct = None;
            self.resolution_changed = true;
        }
        self.configured = true;
    }
}

impl ShmHandler for WaylandLayerContext {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm_state
    }
}

impl Dispatch<wl_buffer::WlBuffer, usize> for WaylandLayerContext {
    /// `wl_buffer.release`: сервер закончил читать буфер — слот снова готов к записи.
    /// Разрыв событий гарантирован roundtrip'ом в основном цикле демона.
    ///
    /// Слот размораживается только если release пришёл от буфера, который числится
    /// в текущем `direct` (сравнение по ObjectId): release буфера прошлого
    /// презентера не должен открывать слоты нового.
    fn event(
        state: &mut Self,
        _proxy: &wl_buffer::WlBuffer,
        event: wl_buffer::Event,
        _udata: &usize,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_buffer::Event::Release = event {
            if let Some(direct) = state.direct.as_mut() {
                match direct.buffers.iter_mut().find(|(buffer, _)| *buffer == *_proxy) {
                    Some((_, released)) => *released = true,
                    // Чужой (уже заменённый) буфер: игнорируем, слоты чужие
                    None => {}
                }
            }
        }
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for WaylandLayerContext {
    /// У `wl_shm_pool` нет событий: заглушка нужна для Dispatch-конструктора
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm_pool::WlShmPool,
        _event: wl_shm_pool::Event,
        _udata: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

delegate_compositor!(WaylandLayerContext);
delegate_output!(WaylandLayerContext);
delegate_shm!(WaylandLayerContext);
delegate_layer!(WaylandLayerContext);
delegate_registry!(WaylandLayerContext);

impl ProvidesRegistryState for WaylandLayerContext {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

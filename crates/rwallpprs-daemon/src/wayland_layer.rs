//! Integration with the Wayland `zwlr_layer_shell_v1` protocol for the **Niri** compositor.

use anyhow::Context as _;
use rustix::fs::fstat;
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};
use rwallpprs_ipc::{OutputStatus, ShmBufferHeader};
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
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_buffer, wl_output, wl_shm, wl_shm_pool, wl_surface},
    Connection, Dispatch, QueueHandle,
};

/// Zero-copy direct presenter: imports the engine's `memfd` into a `wl_shm_pool`
/// so frames reach the Wayland compositor without any intermediate `memcpy`.
pub struct DirectShmPresent {
    /// Kept alive while buffers from it remain attached to the surface.
    #[allow(dead_code)]
    pool: wl_shm_pool::WlShmPool,
    /// `(wl_buffer, released)`: `true` means the compositor released the slot for reuse.
    buffers: Vec<(wl_buffer::WlBuffer, bool)>,
    width: u32,
    height: u32,
    /// Raw descriptor and dimensions of the backing `memfd`.
    source_fd: RawFd,
    /// Kernel file identity (`st_dev`, `st_ino`) used to detect descriptor reuse across engine switches.
    source_dev: u64,
    source_ino: u64,
    /// Daemon mapping of the `memfd` used when decoding video frames directly into SHM slots.
    shm_ptr: NonNull<u8>,
    slot_size_bytes: usize,
}

impl DirectShmPresent {
    /// Imports `direct_fd` as a `wl_shm_pool` and creates two `wl_buffer` objects for slots 0 and 1
    /// (`XRGB8888`, stride `width * 4`).
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

        let raw_ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                total_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                borrowed_fd,
                0,
            )
            .context("rustix::mm::mmap failed in the daemon for direct present")?
        };
        let shm_ptr = match NonNull::new(raw_ptr.cast::<u8>()) {
            Some(p) => p,
            None => {
                unsafe {
                    let _ = munmap(raw_ptr.cast(), total_size);
                }
                anyhow::bail!("mmap returned a null pointer");
            }
        };

        let stat = match fstat(borrowed_fd) {
            Ok(stat) => stat,
            Err(e) => {
                unsafe {
                    let _ = munmap(shm_ptr.as_ptr().cast(), total_size);
                }
                return Err(e).context("rustix::fs::fstat failed on the direct present source");
            }
        };

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
                (buffer, true)
            })
            .collect::<Vec<_>>();

        if write_header {
            unsafe {
                std::ptr::write(shm_ptr.as_ptr().cast::<ShmBufferHeader>(), header);
            }
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

    /// Returns a mutable slice to slot `0` or `1` if the compositor has released that buffer.
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
        for (buffer, _) in &self.buffers {
            buffer.destroy();
        }
        self.pool.destroy();

        let total_size = ShmBufferHeader::HEADER_SIZE + self.slot_size_bytes * 2;
        unsafe {
            let _ = munmap(self.shm_ptr.as_ptr().cast(), total_size);
        }
    }
}

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

    /// Queries the dimensions of the first active Wayland output mode, if reported by the compositor.
    fn primary_output_dimensions(&self) -> Option<(u32, u32)> {
        for output in self.output_state.outputs() {
            if let Some(info) = self.output_state.info(&output) {
                if let Some(mode) = info
                    .modes
                    .iter()
                    .find(|m| m.current)
                    .or_else(|| info.modes.first())
                {
                    let (w, h) = mode.dimensions;
                    if w > 0 && h > 0 {
                        return Some((w as u32, h as u32));
                    }
                }
            }
        }
        None
    }

    /// Returns telemetry for all connected Wayland outputs (falling back to a synthetic entry
    /// if the compositor has not advertised `wl_output` details yet).
    pub fn query_outputs(&self, surface_visible: bool, current_fps: f32) -> Vec<OutputStatus> {
        let mut list = Vec::new();
        for output in self.output_state.outputs() {
            if let Some(info) = self.output_state.info(&output) {
                let name = info
                    .name
                    .clone()
                    .unwrap_or_else(|| "Niri-Primary-Output".to_string());
                let (w, h, refresh_mhz) = info
                    .modes
                    .iter()
                    .find(|m| m.current)
                    .or_else(|| info.modes.first())
                    .map(|m| {
                        (
                            (m.dimensions.0.max(0) as u32).max(self.width),
                            (m.dimensions.1.max(0) as u32).max(self.height),
                            m.refresh_rate.max(0) as u32,
                        )
                    })
                    .unwrap_or((self.width, self.height, 60_000));
                list.push(OutputStatus {
                    name,
                    width: self.width.max(w.min(self.width)),
                    height: self.height.max(h.min(self.height)),
                    refresh_mhz,
                    surface_visible,
                    current_fps,
                });
            }
        }
        if list.is_empty() {
            list.push(OutputStatus {
                name: "Niri-Primary-Output".into(),
                width: self.width,
                height: self.height,
                refresh_mhz: 60_000,
                surface_visible,
                current_fps,
            });
        }
        list
    }

    pub fn present_frame(&mut self, pixels_bgra: &[u8]) -> anyhow::Result<()> {
        if !self.configured {
            anyhow::bail!("present_frame: surface is not configured by the compositor yet");
        }
        let width = self.width;
        let height = self.height;
        let stride = width as i32 * 4;
        let expected_bytes = (width as usize) * (height as usize) * 4;

        if pixels_bgra.len() < expected_bytes {
            anyhow::bail!(
                "present_frame: frame is {} bytes < expected {} ({}x{})",
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

    pub fn init_direct_shm(
        &mut self,
        qh: &QueueHandle<Self>,
        direct_fd: RawFd,
        width: u32,
        height: u32,
        write_header: bool,
    ) -> anyhow::Result<()> {
        if !self.configured {
            anyhow::bail!("init_direct_shm: surface is not configured by the compositor");
        }
        if self.direct.is_some() {
            anyhow::bail!("init_direct_shm: direct mode is already active");
        }
        let shm = self.shm_state.wl_shm().clone();
        self.direct = Some(DirectShmPresent::init(
            &shm,
            qh,
            direct_fd,
            width,
            height,
            write_header,
        )?);
        Ok(())
    }

    pub fn reset_direct_if_stale(&mut self, source: Option<(RawFd, u32, u32)>) {
        let stale = match (&self.direct, source) {
            (Some(d), Some((fd, w, h))) => {
                (d.source_fd, d.width, d.height) != (fd, w, h)
                    || !file_identity_matches(fd, d.source_dev, d.source_ino)
            }
            (Some(_), None) => true,
            (None, _) => false,
        };
        if stale {
            self.direct = None;
        }
    }

    pub fn direct_slot_free(&self, slot: u8) -> bool {
        match &self.direct {
            Some(direct) => direct
                .buffers
                .get((slot & 1) as usize)
                .map_or(true, |(_, released)| *released),
            None => true,
        }
    }

    pub fn present_shm_slot(&mut self, slot: usize) -> anyhow::Result<bool> {
        if !self.configured {
            anyhow::bail!("present_shm_slot: surface is not configured by the compositor");
        }
        let slot = slot & 1;

        match (&self.layer_surface, self.direct.as_mut()) {
            (Some(layer_surface), Some(direct)) => {
                if !direct.buffers[slot].1 {
                    return Ok(false);
                }
                let wl_surf = layer_surface.wl_surface();
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

    pub fn direct_slot_slice(&mut self, slot: usize) -> Option<&mut [u8]> {
        let direct = self.direct.as_mut()?;
        unsafe { direct.slot_slice_mut(slot) }
    }

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
        // Per zwlr_layer_shell_v1 protocol, (0, 0) means the compositor lets the client choose
        // its surface dimensions; use the active wl_output mode or preserve current dimensions.
        let (new_w, new_h) = if configure.new_size.0 > 0 && configure.new_size.1 > 0 {
            (configure.new_size.0.max(64), configure.new_size.1.max(64))
        } else if let Some((ow, oh)) = self.primary_output_dimensions() {
            (ow, oh)
        } else {
            (self.width, self.height)
        };

        if new_w != self.width || new_h != self.height {
            self.width = new_w;
            self.height = new_h;
            self.pool = None;
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
                if let Some((_, released)) = direct
                    .buffers
                    .iter_mut()
                    .find(|(buffer, _)| *buffer == *_proxy)
                {
                    *released = true;
                }
            }
        }
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for WaylandLayerContext {
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

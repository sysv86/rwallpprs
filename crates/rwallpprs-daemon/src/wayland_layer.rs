//! Integration with the Wayland `zwlr_layer_shell_v1` protocol for the **Niri** compositor.

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

/// Zero-copy direct present: the `wl_shm` pool is created over the same `memfd`
/// the engine renders into (a Rust binary over the SHM protocol, video through the
/// daemon's own memfd ring). Frames reach the compositor with **not a single
/// memcpy**; the main loop falls back to the copying `present_frame` when direct
/// mode is unavailable.
pub struct DirectShmPresent {
    /// Kept alive while buffers from it are still active on the surface
    #[allow(dead_code)]
    pool: wl_shm_pool::WlShmPool,
    /// `(wl_buffer, released)`: `true` means the server returned the buffer and the slot can be filled
    buffers: Vec<(wl_buffer::WlBuffer, bool)>,
    width: u32,
    height: u32,
    /// fd and dimensions of the source memfd the pool was created over. Needed for the
    /// consistency check in the daemon main loop: after an engine switch (SetVideo /
    /// SetRustBin / SetEmbeddedBin) a stale presenter would otherwise keep showing the
    /// compositor buffers from someone else's memfd.
    source_fd: RawFd,
    /// File identity of the source (dev/ino from fstat): the kernel reuses fd numbers
    /// after close, so a fd number alone cannot tell an old memfd from a new one.
    source_dev: u64,
    source_ino: u64,
    /// Second daemon mmap of the same memfd (pointing exactly at the engine slots).
    /// Used to write frames (video) straight into compositor memory without copies.
    shm_ptr: NonNull<u8>,
    slot_size_bytes: usize,
}

impl DirectShmPresent {
    /// Imports the engine memfd (`direct_fd`) as a `wl_shm` pool and creates one
    /// `wl_buffer` per `ShmBufferHeader` slot offset (0 and 1), stride `width*4`,
    /// format `XRGB8888`. `udata=slot` ties release events to slots.
    ///
    /// `write_header=true` is for the daemon video memfd ring: the header only
    /// appears here. For a Rust runner memfd pass `false`: `spawn_internal`
    /// already wrote the header.
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

        // mmap first: on failure there is no wl_shm_pool or wl_buffer yet, so
        // there is nothing to clean up and the error path stays transparent
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
                unsafe { let _ = munmap(raw_ptr.cast(), total_size); }
                anyhow::bail!("mmap returned a null pointer");
            }
        };

        // File identity of the source: the main loop needs dev/ino to tell this
        // memfd apart from a new one reusing the same fd number
        let stat = match fstat(borrowed_fd) {
            Ok(stat) => stat,
            Err(e) => {
                unsafe { let _ = munmap(shm_ptr.as_ptr().cast(), total_size); }
                return Err(e).context("rustix::fs::fstat failed on the direct present source");
            }
        };

        // The daemon maps the same memfd so it can write frames straight into the slots
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
                // The server has shown nothing yet, so both slots count as free
                (buffer, true)
            })
            .collect::<Vec<_>>();

        // The header is rewritten only where it does not exist yet: for the daemon
        // video memfd it appears right here; the Rust runner memfd is already laid out.
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

    /// Direct `&mut` slice of a memfd slot (0/1) for the engine to write a frame into — zero copies.
    /// `None` if the slot is still held by the compositor (it must not be overwritten).
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
        // Explicitly send wl_buffer.destroy for both buffers and wl_shm_pool.destroy
        // so the compositor drops its reference to the memfd memory; with the
        // wayland-client 0.31 typed proxies destroy is a generated request.
        for (buffer, _) in &self.buffers {
            buffer.destroy();
        }
        self.pool.destroy();

        // Release the daemon mmap: the compositor holds its own mapping through its
        // own fd, so only this process's virtual memory is dropped here.
        let total_size = ShmBufferHeader::HEADER_SIZE + self.slot_size_bytes * 2;
        unsafe {
            let _ = munmap(self.shm_ptr.as_ptr().cast(), total_size);
        }
    }
}

/// File identity check: compares dev/ino via fstat. fd numbers are reused after
/// close, so the number alone guarantees nothing.
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
    /// Zero-copy direct presenter over the engine memfd; `None` means the copying `present_frame` path
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

    /// Brings up zero-copy direct present: the `direct_fd` memfd is imported into `wl_shm`
    /// as a pool, and the compositor displays exactly the daemon memory the engine renders into.
    /// Return `Err` tells the main loop to fall back to the copying `present_frame`.
    ///
    /// `write_header=true` is only for the daemon video ring (the header appears
    /// here); for a Rust runner memfd pass `false` (the header is already written).
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
            &shm, qh, direct_fd, width, height, write_header,
        )?);
        Ok(())
    }

    /// Keeps the direct presenter consistent with the current engine: if the
    /// presenter points at a foreign or unavailable source (`None`) or its fd no
    /// longer matches the current one (`Some`), it is dropped. Called with the fd of
    /// the current `VideoDirectRing` for video, and with `runner.shm_raw_fd()` for a
    /// Rust binary.
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

    /// Whether the direct presenter slot is free for the next frame (the compositor
    /// already sent release). `true` when direct is absent: flow control must not gate
    /// a path where direct does not exist, since the copying fallback covers it.
    pub fn direct_slot_free(&self, slot: u8) -> bool {
        match &self.direct {
            Some(direct) => direct
                .buffers
                .get((slot & 1) as usize)
                .map_or(true, |(_, released)| *released),
            None => true,
        }
    }

    /// Zero-copy presentation of a slot: attach the direct `wl_buffer` over the memfd + full damage.
    /// `Ok(true)` means the frame reached the screen; `Ok(false)` means direct is unavailable or the
    /// slot is busy (the server has not returned the buffer yet), so the main loop falls back to `present_frame`.
    pub fn present_shm_slot(&mut self, slot: usize) -> anyhow::Result<bool> {
        if !self.configured {
            anyhow::bail!("present_shm_slot: surface is not configured by the compositor");
        }
        let slot = slot & 1;

        // Split field borrows: layer_surface (immutable) and direct (mutable)
        match (&self.layer_surface, self.direct.as_mut()) {
            (Some(layer_surface), Some(direct)) => {
                if !direct.buffers[slot].1 {
                    // Slot still held by the compositor (no release yet) — use the copying path
                    return Ok(false);
                }
                let wl_surf = layer_surface.wl_surface();
                // Attach the memfd memory directly so pixels reach the compositor with no copies
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

    /// Hands out a `&mut` slice of a memfd slot (0/1) for the engine to write a frame
    /// into directly (e.g. `video.decode_next_frame_into`). `None` means the copying path.
    pub fn direct_slot_slice(&mut self, slot: usize) -> Option<&mut [u8]> {
        let direct = self.direct.as_mut()?;
        unsafe { direct.slot_slice_mut(slot) }
    }

    /// Marks the wallpaper as hidden from drawing without detaching the surface buffer:
    /// actually detaching a wl_buffer from a layer surface makes Niri raise a protocol
    /// error and kills the layer (see the call site in main.rs).
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
            // The direct presenter targets the previous memfd layout, so fall back to
            // copying; it will be lazily recreated on the engine's next frame
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
    /// `wl_buffer.release`: the server finished reading the buffer, so the slot is ready again.
    /// Event ordering is guaranteed by the roundtrip in the daemon main loop.
    ///
    /// The slot is unfrozen only when the release came from a buffer currently listed in
    /// `direct` (compared by ObjectId): a release from a previous presenter must not open
    /// slots of the new one.
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
                    // Foreign (already replaced) buffer: ignore, those slots are not ours
                    None => {}
                }
            }
        }
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for WaylandLayerContext {
    /// `wl_shm_pool` has no events: this stub only satisfies the Dispatch trait
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

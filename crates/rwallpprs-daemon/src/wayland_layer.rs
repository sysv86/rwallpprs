//! Интеграция с протоколом Wayland `zwlr_layer_shell_v1` для композитора **Niri**.

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
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_output, wl_shm, wl_surface},
    Connection, QueueHandle,
};

pub struct WaylandLayerContext {
    pub registry_state: RegistryState,
    pub output_state: OutputState,
    pub compositor_state: CompositorState,
    pub shm_state: Shm,
    pub layer_shell: LayerShell,
    pub pool: Option<SlotPool>,
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

    pub fn hide_surface_for_powersave(&mut self) {
        if self.surface_hidden {
            return;
        }
        if let Some(layer_surface) = &self.layer_surface {
            let wl_surf = layer_surface.wl_surface();
            wl_surf.attach(None, 0, 0);
            wl_surf.commit();
            self.surface_hidden = true;
        }
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

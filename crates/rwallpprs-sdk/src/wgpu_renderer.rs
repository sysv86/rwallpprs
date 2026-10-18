//! Hardware GPU renderer for Rust wallpapers built on [`wgpu`] (Vulkan / GL / Metal).
//!
//! Lets you write **WGSL** fragment shaders that execute on the GPU in headless mode
//! (offscreen `Bgra8Unorm` texture), automatically receive **Niri** compositor state
//! through the [`NiriShaderUniforms`] uniform buffer, and write finished pixels
//! (`BGRA8` / `ARGB8888`) straight into the daemon `memfd` shared memory.
//!
//! For complex 2D scenes (thousands of lines/circles/polygons, instancing, custom
//! pipelines, multiple passes) use the scene API: the [`GpuSceneWallpaper`] trait,
//! the [`GpuScene`] frame context and the [`SceneWallpaperAdapter`] adapter with the
//! `run_scene_wallpaper!` macro.

use crate::{Canvas, FrameContext, RustWallpaperMode, Wallpaper};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::mpsc::TryRecvError;

/// Offscreen texture format (`Bgra8Unorm`) that scene wallpapers render into.
///
/// Matches the Wayland `ARGB8888` / `XRGB8888` byte order on little-endian, so the
/// finished frame can be memcpy'd straight into a [`Canvas`] without conversion.
pub const SCENE_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// Unpacks a row-padded mapped staging buffer (256-byte row alignment) into
/// [`Canvas`] pixels, ignoring each row's padding.
fn copy_rows_to_canvas(
    mapped: &[u8],
    canvas: &mut Canvas<'_>,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) {
    let dest_bytes: &mut [u8] = bytemuck::cast_slice_mut(canvas.pixels_mut());
    let row_bytes = (width as usize) * 4;
    if row_bytes == 0 {
        return;
    }

    let padded_stride = padded_bytes_per_row as usize;
    let rows = (height as usize).min(dest_bytes.len() / row_bytes);

    if padded_stride == row_bytes {
        let total = row_bytes * rows;
        dest_bytes[..total].copy_from_slice(&mapped[..total]);
    } else {
        for y in 0..rows {
            let src_off = y * padded_stride;
            let dst_off = y * row_bytes;
            dest_bytes[dst_off..dst_off + row_bytes]
                .copy_from_slice(&mapped[src_off..src_off + row_bytes]);
        }
    }
}

/// Number of staging buffers in the readback pool.
///
/// Two are enough for the "GPU draws frame `N` while the CPU reads frame `N-1`"
/// pipeline; a third adds slack so the CPU does not run into busy buffers during
/// short GPU latency spikes.
const READBACK_SLOTS: usize = 3;

/// State of a single staging buffer in the readback pool.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// Buffer is free (not mapped) and ready for a copy of the next frame.
    Free,
    /// Copy submitted to the GPU, `map_async` waits for the submission to finish.
    InFlight,
    /// `map_async` completed successfully — data is ready to copy into [`Canvas`].
    Mapped,
}

/// One staging buffer of the pool together with its readback state.
struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: SlotState,
    /// Submission the buffer was filled in (for a targeted `WaitForSubmissionIndex`).
    submission: Option<wgpu::SubmissionIndex>,
    /// Receiver for the `map_async` result of the slot's current use.
    rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// Pool of staging buffers with double (triple) buffering of the GPU→CPU readback.
///
/// Instead of a synchronous "`submit` → `Maintain::Wait` → `unmap`" pair per frame,
/// the pool holds several buffers with a state slot each. On frame `N` the texture
/// copy goes into a free slot while the already computed frame `N-1` is handed to
/// [`Canvas`]: the GPU keeps working on `N` while the CPU picks up `N-1`. As a
/// result `device.poll(Maintain::Wait)` is no longer called on the submission that
/// was just sent.
///
/// The price is one frame of output latency (irrelevant for wallpapers). The first
/// frame and the frame after a resize are delivered synchronously so the [`Canvas`]
/// is never left empty.
struct StagingPool {
    slots: Vec<ReadbackSlot>,
    /// Slot indices in submission order — the queue for delivery into [`Canvas`].
    pending: VecDeque<usize>,
    /// Cache of the free-slot choice (round-robin).
    cursor: usize,
    /// Aligned (256-byte) row length of the staging buffer.
    bytes_per_row: u32,
    /// How many frames have been delivered into [`Canvas`] (needed for a synchronous first frame).
    delivered: u64,
}

impl StagingPool {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mut pool = Self {
            slots: Vec::with_capacity(READBACK_SLOTS),
            pending: VecDeque::with_capacity(READBACK_SLOTS),
            cursor: 0,
            bytes_per_row: 0,
            delivered: 0,
        };
        pool.recreate(device, width, height);
        pool
    }

    /// Rebuilds the pool for a new frame size and resets the pipeline: the next frame
    /// is delivered synchronously again (see [`Self::finish_frame`]).
    fn recreate(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Unmap mapped buffers first, then release them.
        for slot in &mut self.slots {
            if slot.state == SlotState::Mapped {
                slot.buffer.unmap();
            }
        }
        self.slots.clear();
        self.pending.clear();
        self.delivered = 0;
        self.cursor = 0;

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = (width * 4).div_ceil(align) * align;
        self.bytes_per_row = padded;

        for i in 0..READBACK_SLOTS {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("rwallpprs-staging-readback-{i}")),
                size: (padded as u64) * (height as u64),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.slots.push(ReadbackSlot {
                buffer,
                state: SlotState::Free,
                submission: None,
                rx: None,
            });
        }
    }

    /// Aligned row length to pass to `copy_texture_to_buffer`.
    fn bytes_per_row(&self) -> u32 {
        self.bytes_per_row
    }

    /// Returns the slot buffer to write the frame texture copy into.
    fn slot_buffer(&self, idx: usize) -> &wgpu::Buffer {
        &self.slots[idx].buffer
    }

    /// Picks a free slot for the copy of the new frame.
    ///
    /// If every slot is busy (the GPU is chronically behind), it force-delivers the
    /// oldest frame, blocking on that submission rather than the one just sent.
    fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        canvas: &mut Canvas<'_>,
    ) -> Result<usize> {
        if self.pending.len() >= self.slots.len() {
            self.deliver_oldest(device, canvas)?;
        }
        Ok(self.acquire_free())
    }

    /// Registers `map_async` for the submission just sent and hands an already
    /// computed older frame to [`Canvas`] (in steady state, `N-1`).
    fn finish_frame(
        &mut self,
        device: &wgpu::Device,
        canvas: &mut Canvas<'_>,
        idx: usize,
        submission: wgpu::SubmissionIndex,
    ) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        self.slots[idx]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
        self.slots[idx].rx = Some(rx);
        self.slots[idx].submission = Some(submission);
        self.slots[idx].state = SlotState::InFlight;
        self.pending.push_back(idx);

        // The first frame (and the frame after a resize) is delivered synchronously so the
        // `Canvas` is never empty. After that the pipeline stays one frame deep: deliver `N-1`
        // without touching the fresh `N`, which is exactly what overlaps GPU and CPU work.
        if self.pending.len() > 1 || self.delivered == 0 {
            self.deliver_oldest(device, canvas)?;
        }
        Ok(())
    }

    fn acquire_free(&mut self) -> usize {
        for i in 0..self.slots.len() {
            let idx = (self.cursor + i) % self.slots.len();
            if self.slots[idx].state == SlotState::Free {
                self.cursor = (idx + 1) % self.slots.len();
                return idx;
            }
        }
        // This should be unreachable: `begin_frame` frees a slot when the pool fills up.
        self.cursor = (self.cursor + 1) % self.slots.len();
        self.cursor
    }

    /// Delivers the oldest queued frame into [`Canvas`] and frees its slot.
    fn deliver_oldest(&mut self, device: &wgpu::Device, canvas: &mut Canvas<'_>) -> Result<()> {
        let idx = match self.pending.front() {
            Some(&idx) => idx,
            None => return Ok(()),
        };

        // Cheap non-blocking poll first: usually frame `N-1` is already done.
        let mut ready = self.poll_slot(idx)?;
        if !ready {
            // Wait for this frame's submission (`N-1`) rather than the newest one (`N`), so
            // the GPU keeps working on `N` while the CPU picks up `N-1`.
            match self.slots[idx].submission.take() {
                Some(sub) => device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub)),
                None => device.poll(wgpu::Maintain::Wait),
            };
            ready = self.poll_slot(idx)?;
            if !ready {
                // Safety net for backends without targeted wait support.
                device.poll(wgpu::Maintain::Wait);
                ready = self.poll_slot(idx)?;
            }
        }

        if !ready {
            anyhow::bail!("the readback staging buffer never mapped");
        }

        {
            let slice = self.slots[idx].buffer.slice(..);
            let mapped = slice.get_mapped_range();
            copy_rows_to_canvas(
                &mapped,
                canvas,
                canvas.width,
                canvas.height,
                self.bytes_per_row,
            );
        }
        self.slots[idx].buffer.unmap();
        self.slots[idx].state = SlotState::Free;
        self.slots[idx].submission = None;
        self.slots[idx].rx = None;
        self.pending.pop_front();
        self.delivered += 1;
        Ok(())
    }

    /// Non-blocking check: has the slot's `map_async` finished?
    fn poll_slot(&mut self, idx: usize) -> Result<bool> {
        if self.slots[idx].state == SlotState::Mapped {
            return Ok(true);
        }
        if self.slots[idx].state != SlotState::InFlight {
            return Ok(false);
        }
        let recv = self.slots[idx].rx.as_ref().map(|rx| rx.try_recv());
        match recv {
            Some(Ok(Ok(()))) => {
                self.slots[idx].state = SlotState::Mapped;
                Ok(true)
            }
            Some(Ok(Err(e))) => {
                anyhow::bail!("staging buffer map_async failed: {e}")
            }
            Some(Err(TryRecvError::Empty)) => Ok(false),
            Some(Err(TryRecvError::Disconnected)) => {
                anyhow::bail!("staging buffer map_async callback lost")
            }
            None => Ok(false),
        }
    }
}

/// Layout of the uniform buffer (`@group(0) @binding(0) var<uniform> u: NiriShaderUniforms;`)
/// available inside any wallpaper WGSL shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NiriShaderUniforms {
    /// Screen resolution in pixels `(width, height)`.
    pub resolution: [f32; 2],
    /// Seconds since startup.
    pub time: f32,
    /// Frame delta time in seconds.
    pub dt: f32,
    /// Horizontal scroll offset of the Niri strip.
    pub niri_scroll_offset: f32,
    /// Index of the active Niri workspace.
    pub niri_workspace_idx: u32,
    /// `1.0` if Niri Overview (`Mod+O`) is open, otherwise `0.0`.
    pub niri_overview_open: f32,
    /// Number of visible windows on the active Niri workspace.
    pub niri_windows_count: u32,
}

/// High-level trait for hardware GPU wallpapers driven by a WGSL shader.
pub trait GpuShaderWallpaper: Sized + 'static {
    /// Initialize wallpaper parameters.
    fn init(args: &[String]) -> Self;

    /// WGSL shader source (must provide either both `vs_main` and `fs_main`, or only
    /// `fs_main(in: VertexOutput) -> @location(0) vec4<f32>` when the default fullscreen
    /// triangle vertex shader is used).
    fn wgsl_fragment_source(&self) -> &str;

    /// Run mode (`Animated`, `Static`, `ReactiveNiri`).
    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    /// CPU fallback for environments without a GPU/Vulkan.
    fn fallback_cpu_render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>);
}

pub struct WgpuHeadlessPipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    target_texture: wgpu::Texture,
    target_view: wgpu::TextureView,
    pool: StagingPool,
    width: u32,
    height: u32,
}

impl WgpuHeadlessPipeline {
    pub fn new(fragment_wgsl: &str, width: u32, height: u32) -> Result<Self> {
        pollster::block_on(Self::new_async(fragment_wgsl, width, height))
    }

    async fn new_async(fragment_wgsl: &str, width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("No compatible Vulkan/GL adapter found for wgpu")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .context("Failed to open a logical wgpu::Device")?;

        // Assemble the full shader module with the built-in fullscreen triangle vertex shader
        let full_wgsl = format!(
            r#"
struct NiriShaderUniforms {{
    resolution: vec2<f32>,
    time: f32,
    dt: f32,
    niri_scroll_offset: f32,
    niri_workspace_idx: u32,
    niri_overview_open: f32,
    niri_windows_count: u32,
}};

@group(0) @binding(0)
var<uniform> u: NiriShaderUniforms;

struct VertexOutput {{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {{
    var out: VertexOutput;
    let x = f32((in_vertex_index << 1u) & 2u);
    let y = f32(in_vertex_index & 2u);
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}}

{fragment_wgsl}
"#
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rwallpprs-wgsl-shader"),
            source: wgpu::ShaderSource::Wgsl(full_wgsl.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("niri-uniform-buffer"),
            size: std::mem::size_of::<NiriShaderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("niri-bind-group-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("niri-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rwallpprs-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rwallpprs-render-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (target_texture, target_view) = Self::create_render_targets(&device, width, height);
        let pool = StagingPool::new(&device, width, height);

        Ok(Self {
            device,
            queue,
            pipeline,
            uniform_buffer,
            bind_group,
            target_texture,
            target_view,
            pool,
            width,
            height,
        })
    }

    fn create_render_targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rwallpprs-offscreen-bgra"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        (texture, view)
    }

    pub fn resize_if_needed(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let (tex, view) = Self::create_render_targets(&self.device, width, height);
        self.target_texture = tex;
        self.target_view = view;
        self.pool.recreate(&self.device, width, height);
        self.width = width;
        self.height = height;
    }

    /// Executes the WGSL shader on the GPU and copies the result straight into the zero-copy [`Canvas`] (`memfd`).
    pub fn render_to_canvas(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> Result<()> {
        self.resize_if_needed(ctx.width, ctx.height);

        let uniforms = NiriShaderUniforms {
            resolution: [ctx.width as f32, ctx.height as f32],
            time: ctx.time_secs as f32,
            dt: ctx.dt_secs,
            niri_scroll_offset: ctx.niri.strip_scroll_offset as f32,
            niri_workspace_idx: ctx.niri.active_workspace_idx,
            niri_overview_open: if ctx.niri.overview_open { 1.0 } else { 0.0 },
            niri_windows_count: ctx.niri.visible_windows_count,
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Slot that receives this frame's copy; the pool may also deliver frame N-1 to the canvas.
        let slot = self.pool.begin_frame(&self.device, canvas)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rwallpprs-frame-encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rwallpprs-fullscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: self.pool.slot_buffer(slot),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.pool.bytes_per_row()),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let submission = self.queue.submit(Some(encoder.finish()));
        self.pool.finish_frame(&self.device, canvas, slot, submission)
    }
}

/// Adapter turning any [`GpuShaderWallpaper`] into a standard [`Wallpaper`], with automatic
/// CPU fallback when Vulkan/GL is unavailable.
pub struct ShaderWallpaperAdapter<S: GpuShaderWallpaper> {
    inner: S,
    gpu: Option<WgpuHeadlessPipeline>,
    gpu_init_attempted: bool,
}

impl<S: GpuShaderWallpaper> Wallpaper for ShaderWallpaperAdapter<S> {
    fn init(args: &[String]) -> Self {
        let inner = S::init(args);
        Self {
            inner,
            gpu: None,
            gpu_init_attempted: false,
        }
    }

    fn default_mode(&self) -> RustWallpaperMode {
        self.inner.default_mode()
    }

    fn render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> bool {
        if !self.gpu_init_attempted {
            self.gpu_init_attempted = true;
            match WgpuHeadlessPipeline::new(
                self.inner.wgsl_fragment_source(),
                ctx.width,
                ctx.height,
            ) {
                Ok(pipeline) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU (Vulkan/GL) shader pipeline initialized successfully ({}x{})",
                        ctx.width, ctx.height
                    );
                    self.gpu = Some(pipeline);
                }
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU unavailable ({}), using the CPU renderer",
                        e
                    );
                }
            }
        }

        if let Some(gpu) = &mut self.gpu {
            if gpu.render_to_canvas(ctx, canvas).is_ok() {
                return true;
            }
            self.gpu = None;
        }

        self.inner.fallback_cpu_render(ctx, canvas);
        true
    }
}

/// Per-frame context for **scene** GPU wallpapers.
///
/// Unlike [`GpuShaderWallpaper`] (a single fragment WGSL shader), scene wallpapers get
/// full access to [`wgpu::Device`], [`wgpu::Queue`] and [`wgpu::CommandEncoder`], so they
/// can build their own pipelines, vertex/index buffers, instancing (thousands of lines,
/// circles, polygons), multiple render passes and intermediate textures.
///
/// The wallpaper writes its own render passes into [`GpuScene::encoder`], drawing into
/// [`GpuScene::target`] (format [`GpuScene::format`] = [`SCENE_TARGET_FORMAT`]). The SDK
/// copies the result into the [`Canvas`] itself.
pub struct GpuScene<'a> {
    /// Logical `wgpu` device (creating buffers, pipelines, textures, etc.).
    pub device: &'a wgpu::Device,
    /// `wgpu` queue (uploading data into buffers/textures via `write_buffer` / `write_texture`).
    pub queue: &'a wgpu::Queue,
    /// Command encoder for the current frame — wallpapers write their render passes here.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Target offscreen texture for the frame (`Bgra8Unorm`) to draw into.
    pub target: &'a wgpu::TextureView,
    /// Target texture format (`Bgra8Unorm`).
    pub format: wgpu::TextureFormat,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// High-level trait for **scene** hardware wallpapers built on full `wgpu`.
///
/// Use it when a fragment shader is not enough: thousands of geometry instances,
/// custom pipelines, multiple passes, compute shaders preparing data, and so on.
pub trait GpuSceneWallpaper: Sized + 'static {
    /// Initialize wallpaper parameters.
    fn init(args: &[String]) -> Self;

    /// Run mode (`Animated`, `Static`, `ReactiveNiri`).
    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    /// Called by the daemon when the wallpaper sleeps or is hidden (e.g. to pause the simulation).
    fn on_suspend(&mut self) {}

    /// Called by the daemon when the wallpaper resumes.
    fn on_resume(&mut self) {}

    /// Draws one scene frame, writing render passes into `scene.encoder` and drawing into `scene.target`.
    ///
    /// Returns `true` when the frame changed (the daemon updates the desktop). `false` is
    /// allowed for pauses when the content did not change — the daemon then skips the frame.
    fn render_scene(&mut self, ctx: &FrameContext, scene: &mut GpuScene<'_>) -> bool;

    /// CPU fallback for environments without a GPU/Vulkan.
    ///
    /// Required: a minimal image so the wallpaper still works without a GPU.
    fn fallback_cpu_render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>);
}

/// Internal headless pipeline for scene wallpapers: offscreen texture + staging buffer pool
/// for asynchronous readback.
struct WgpuScenePipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target_texture: wgpu::Texture,
    target_view: wgpu::TextureView,
    pool: StagingPool,
    width: u32,
    height: u32,
}

impl WgpuScenePipeline {
    async fn new_async(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("No compatible Vulkan/GL adapter found for wgpu")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .context("Failed to open a logical wgpu::Device")?;

        let (target_texture, target_view) =
            WgpuHeadlessPipeline::create_render_targets(&device, width, height);
        let pool = StagingPool::new(&device, width, height);

        Ok(Self {
            device,
            queue,
            target_texture,
            target_view,
            pool,
            width,
            height,
        })
    }

    fn resize_if_needed(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let (tex, view) = WgpuHeadlessPipeline::create_render_targets(&self.device, width, height);
        self.target_texture = tex;
        self.target_view = view;
        self.pool.recreate(&self.device, width, height);
        self.width = width;
        self.height = height;
    }

    /// Lets the wallpaper draw its scene, then copies the offscreen texture into the [`Canvas`] (`memfd`).
    ///
    /// The copy of frame `N` goes into a free pool slot and is NOT awaited in place: each
    /// frame delivers the already computed `N-1` (`StagingPool::deliver_oldest`).
    fn render_frame<S: GpuSceneWallpaper>(
        &mut self,
        inner: &mut S,
        ctx: &FrameContext,
        canvas: &mut Canvas<'_>,
    ) -> Result<bool> {
        self.resize_if_needed(ctx.width, ctx.height);

        // Slot that receives this frame's copy; the pool may also deliver frame N-1 to the canvas.
        let slot = self.pool.begin_frame(&self.device, canvas)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rwallpprs-scene-encoder"),
            });

        // Hand the wallpaper its own encoder and target; it writes its render passes there.
        let changed = {
            let mut scene = GpuScene {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                target: &self.target_view,
                format: SCENE_TARGET_FORMAT,
                width: self.width,
                height: self.height,
            };
            inner.render_scene(ctx, &mut scene)
        };

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: self.pool.slot_buffer(slot),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.pool.bytes_per_row()),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let submission = self.queue.submit(Some(encoder.finish()));
        self.pool.finish_frame(&self.device, canvas, slot, submission)?;

        Ok(changed)
    }
}

/// Writes the current frame to a PPM (`P6`) for visual inspection without running the daemon.
///
/// The path comes from the `RWALLPPRS_DUMP_PPM` environment variable (see [`SceneWallpaperAdapter`]).
/// [`Canvas`] pixels are stored as `BGRA` (little-endian `ARGB8888`); the PPM gets `RGB` without alpha.
fn write_ppm_dump(path: &str, width: u32, height: u32, pixels: &[u32]) -> std::io::Result<()> {
    let count = (width as usize) * (height as usize);
    let mut buf = Vec::with_capacity(32 + count * 3);
    buf.extend_from_slice(format!("P6\n{} {}\n255\n", width, height).as_bytes());
    for &px in pixels.iter().take(count) {
        buf.push(((px >> 16) & 0xFF) as u8); // R
        buf.push(((px >> 8) & 0xFF) as u8); // G
        buf.push((px & 0xFF) as u8); // B
    }
    std::fs::write(path, buf)
}

/// Adapter turning any [`GpuSceneWallpaper`] into a standard [`Wallpaper`], with lazy GPU
/// initialization and automatic CPU fallback when Vulkan/GL is unavailable.
pub struct SceneWallpaperAdapter<S: GpuSceneWallpaper> {
    inner: S,
    gpu: Option<WgpuScenePipeline>,
    gpu_init_attempted: bool,
    ppm_dump_path: Option<String>,
}

impl<S: GpuSceneWallpaper> Wallpaper for SceneWallpaperAdapter<S> {
    fn init(args: &[String]) -> Self {
        let inner = S::init(args);
        Self {
            inner,
            gpu: None,
            gpu_init_attempted: false,
            ppm_dump_path: std::env::var("RWALLPPRS_DUMP_PPM").ok(),
        }
    }

    fn default_mode(&self) -> RustWallpaperMode {
        self.inner.default_mode()
    }

    fn on_suspend(&mut self) {
        self.inner.on_suspend();
    }

    fn on_resume(&mut self) {
        self.inner.on_resume();
    }

    fn render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> bool {
        if !self.gpu_init_attempted {
            self.gpu_init_attempted = true;
            match pollster::block_on(WgpuScenePipeline::new_async(ctx.width, ctx.height)) {
                Ok(gpu) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU (Vulkan/GL) scene pipeline initialized successfully ({}x{})",
                        ctx.width, ctx.height
                    );
                    self.gpu = Some(gpu);
                }
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU unavailable ({}), using the CPU renderer",
                        e
                    );
                }
            }
        }

        let mut changed = true;

        if let Some(gpu) = &mut self.gpu {
            match gpu.render_frame(&mut self.inner, ctx, canvas) {
                Ok(frame_changed) => changed = frame_changed,
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] Scene GPU pipeline error ({}), falling back to the CPU renderer",
                        e
                    );
                    self.gpu = None;
                }
            }
        }

        // The GPU never initialized or died — draw a minimal CPU image.
        if self.gpu.is_none() {
            self.inner.fallback_cpu_render(ctx, canvas);
            changed = true;
        }

        if let Some(path) = self.ppm_dump_path.take() {
            if let Err(e) = write_ppm_dump(&path, ctx.width, ctx.height, canvas.pixels_mut()) {
                eprintln!("[rwallpprs-sdk] Failed to write the PPM dump '{}': {}", path, e);
            } else {
                self.ppm_dump_path = Some(path);
            }
        }

        changed
    }
}

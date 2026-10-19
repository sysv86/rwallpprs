//! Hardware GPU renderer for Rust wallpapers built on [`wgpu`] (Vulkan / GL).
//!
//! Provides two GPU wallpaper abstractions backed by a unified, lightweight [`WgpuCore`]:
//! - **Fragment Shader Wallpapers ([`GpuShaderWallpaper`])**: Runs WGSL fragment (and optional
//!   vertex) shaders into an offscreen `Bgra8Unorm` render target with automatic [`NiriShaderUniforms`]
//!   upload and zero-conversion readback into the daemon's `wl_shm` shared memory.
//! - **Full 2D/3D Scene Wallpapers ([`GpuSceneWallpaper`])**: Exposes raw [`wgpu::Device`],
//!   [`wgpu::Queue`], and [`wgpu::CommandEncoder`] access via [`GpuScene`] for custom pipelines,
//!   instanced SDF geometry, vertex/index buffers, and multi-pass rendering.
//!
//! ### GPU Subsystem Highlights
//! - **Unified `WgpuCore`**: Eliminates duplicate instance/adapter/device setup between shader
//!   and scene pipelines, disables debug/validation layers by default for minimal driver overhead,
//!   and cascades adapter selection (`LowPower` -> `HighPerformance` -> software fallback).
//! - **Adaptive `ReadbackRing`**: Uses a single synchronous staging buffer for `Static` and
//!   `ReactiveNiri` wallpapers (saving 66% staging memory with zero frame latency), and a
//!   2-slot ping-pong ring for `Animated` wallpapers (saving 33% staging memory while overlapping
//!   GPU rendering of frame `N` with CPU readback of frame `N-1`).
//! - **Zero Frame-1 Black Flash**: Retains the mapped initial frame across the double-buffer
//!   transition so both Wayland SHM slots (`0` and `1`) are always populated with valid pixels.
//! - **Graceful Fault & Suspend Handling**: Captures uncaptured `wgpu` device errors without
//!   panicking (falling back to CPU rendering) and unmaps all staging buffers on `Suspend`.

use crate::{Canvas, FrameContext, RustWallpaperMode, Wallpaper};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, TryRecvError},
    Arc,
};

/// Offscreen texture format (`Bgra8Unorm`) that shader and scene wallpapers render into.
///
/// Matches the Wayland `ARGB8888` / `XRGB8888` byte order on little-endian architectures,
/// allowing finished rows to be copied straight into a [`Canvas`] without channel swizzling.
pub const SCENE_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// Unpacks a 256-byte row-aligned mapped staging buffer into [`Canvas`] pixels.
#[inline]
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
        if mapped.len() >= total {
            dest_bytes[..total].copy_from_slice(&mapped[..total]);
        }
    } else {
        for y in 0..rows {
            let src_off = y * padded_stride;
            let dst_off = y * row_bytes;
            if src_off + row_bytes <= mapped.len() {
                dest_bytes[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&mapped[src_off..src_off + row_bytes]);
            }
        }
    }
}

/// Lifecycle state of a single staging buffer in [`ReadbackRing`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotState {
    /// Unmapped and ready to receive a `copy_texture_to_buffer` command.
    Free,
    /// GPU copy submitted and `map_async` requested; waiting for completion.
    InFlight,
    /// Mapped and ready to copy into [`Canvas`].
    Mapped,
    /// Mapped on the initial frame and already copied to SHM slot 0; retained so
    /// SHM slot 1 also receives valid pixels on frame 1 while frame 1 renders on the GPU.
    MappedRetained,
}

/// A single `MAP_READ | COPY_DST` staging buffer and its async synchronization state.
struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: SlotState,
    submission: Option<wgpu::SubmissionIndex>,
    rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// Lightweight GPU-to-CPU readback ring with adaptive slot allocation.
///
/// - In **synchronous mode** (`Static` / `ReactiveNiri` or `RWALLPPRS_GPU_SYNC_READBACK=1`),
///   allocates **1 staging buffer** and delivers every frame immediately with zero latency.
/// - In **pipelined mode** (`Animated`), allocates **2 staging buffers** and overlaps GPU
///   rendering of frame `N` with CPU readback of frame `N-1` without ever leaving either
///   SHM double-buffer slot uninitialized.
struct ReadbackRing {
    slots: Vec<ReadbackSlot>,
    /// Index of the slot currently holding the previous frame (`InFlight` or `MappedRetained`).
    in_flight: Option<usize>,
    /// 256-byte aligned row stride required by `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`.
    bytes_per_row: u32,
    /// Whether 2-slot async pipelining is active.
    async_pipelined: bool,
}

impl ReadbackRing {
    fn new(device: &wgpu::Device, width: u32, height: u32, async_pipelined: bool) -> Self {
        let mut ring = Self {
            slots: Vec::new(),
            in_flight: None,
            bytes_per_row: 0,
            async_pipelined,
        };
        ring.recreate(device, width, height);
        ring
    }

    /// Reallocates staging buffers for a new target resolution and resets pipeline state.
    fn recreate(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.release_all_mappings(device);
        self.slots.clear();
        self.in_flight = None;

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = (width * 4).div_ceil(align) * align;
        self.bytes_per_row = padded;

        let slot_count = if self.async_pipelined { 2 } else { 1 };
        let byte_size = (padded as u64) * (height.max(1) as u64);

        self.slots.reserve_exact(slot_count);
        for i in 0..slot_count {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("rwallpprs-readback-slot-{i}")),
                size: byte_size,
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

    #[inline]
    fn bytes_per_row(&self) -> u32 {
        self.bytes_per_row
    }

    /// Selects the staging buffer index that should receive the current frame's GPU copy.
    #[inline]
    fn select_write_slot(&self) -> usize {
        if !self.async_pipelined || self.slots.len() < 2 {
            0
        } else {
            match self.in_flight {
                Some(prev) => (prev ^ 1) & 1,
                None => 0,
            }
        }
    }

    #[inline]
    fn slot_buffer(&self, idx: usize) -> &wgpu::Buffer {
        &self.slots[idx].buffer
    }

    /// Submits `map_async` for `write_idx` and delivers a completed frame into `canvas`.
    fn complete_frame(
        &mut self,
        device: &wgpu::Device,
        canvas: &mut Canvas<'_>,
        write_idx: usize,
        submission: wgpu::SubmissionIndex,
    ) -> Result<()> {
        // If `write_idx` somehow still holds a mapping, unmap it before re-mapping.
        if matches!(
            self.slots[write_idx].state,
            SlotState::Mapped | SlotState::MappedRetained
        ) {
            self.slots[write_idx].buffer.unmap();
            self.slots[write_idx].state = SlotState::Free;
        }

        let (tx, rx) = mpsc::channel();
        self.slots[write_idx]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
        self.slots[write_idx].rx = Some(rx);
        self.slots[write_idx].submission = Some(submission);
        self.slots[write_idx].state = SlotState::InFlight;

        if !self.async_pipelined {
            // Synchronous 1-slot path: wait for the current frame and copy straight to Canvas.
            self.wait_and_copy_slot(device, canvas, write_idx, false)?;
            return Ok(());
        }

        match self.in_flight {
            None => {
                // First frame after startup, resize, or resume: deliver synchronously into
                // SHM slot 0, and retain the mapped buffer so Frame 1 can populate SHM slot 1
                // while Frame 1 is rendering on the GPU.
                self.wait_and_copy_slot(device, canvas, write_idx, true)?;
                self.in_flight = Some(write_idx);
            }
            Some(prev_idx) => {
                // Steady state: GPU is now rendering `write_idx` in the background while
                // we copy `prev_idx` (either `MappedRetained` from Frame 0 or `InFlight`
                // from Frame N-1) into `canvas`.
                self.wait_and_copy_slot(device, canvas, prev_idx, false)?;
                self.in_flight = Some(write_idx);
            }
        }

        Ok(())
    }

    /// Waits for `idx` to be mapped (if not already mapped), copies its pixels into `canvas`,
    /// and either unmaps the slot or retains the mapping for the Frame-0 -> Frame-1 transition.
    fn wait_and_copy_slot(
        &mut self,
        device: &wgpu::Device,
        canvas: &mut Canvas<'_>,
        idx: usize,
        retain_mapping: bool,
    ) -> Result<()> {
        if self.slots[idx].state != SlotState::MappedRetained {
            let mut ready = self.poll_slot(idx)?;
            if !ready {
                match self.slots[idx].submission.take() {
                    Some(sub) => device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub)),
                    None => device.poll(wgpu::Maintain::Wait),
                };
                ready = self.poll_slot(idx)?;
                if !ready {
                    device.poll(wgpu::Maintain::Wait);
                    ready = self.poll_slot(idx)?;
                }
            }
            if !ready {
                anyhow::bail!("readback staging buffer slot {idx} failed to map");
            }
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

        if retain_mapping {
            self.slots[idx].state = SlotState::MappedRetained;
        } else {
            self.slots[idx].buffer.unmap();
            self.slots[idx].state = SlotState::Free;
            self.slots[idx].submission = None;
            self.slots[idx].rx = None;
        }

        Ok(())
    }

    /// Non-blocking check for `map_async` completion on `idx`.
    fn poll_slot(&mut self, idx: usize) -> Result<bool> {
        match self.slots[idx].state {
            SlotState::Mapped | SlotState::MappedRetained => return Ok(true),
            SlotState::Free => return Ok(false),
            SlotState::InFlight => {}
        }

        let recv = self.slots[idx].rx.as_ref().map(|rx| rx.try_recv());
        match recv {
            Some(Ok(Ok(()))) => {
                self.slots[idx].state = SlotState::Mapped;
                Ok(true)
            }
            Some(Ok(Err(e))) => anyhow::bail!("staging buffer map_async error: {e}"),
            Some(Err(TryRecvError::Empty)) => Ok(false),
            Some(Err(TryRecvError::Disconnected)) => {
                anyhow::bail!("staging buffer map_async channel disconnected")
            }
            None => Ok(false),
        }
    }

    /// Drains any in-flight GPU work and unmaps all staging buffers when suspended.
    fn release_all_mappings(&mut self, device: &wgpu::Device) {
        let any_in_flight = self
            .slots
            .iter()
            .any(|s| s.state == SlotState::InFlight);
        if any_in_flight {
            device.poll(wgpu::Maintain::Wait);
        }
        for idx in 0..self.slots.len() {
            let _ = self.poll_slot(idx);
            if matches!(
                self.slots[idx].state,
                SlotState::Mapped | SlotState::MappedRetained | SlotState::InFlight
            ) {
                self.slots[idx].buffer.unmap();
            }
            self.slots[idx].state = SlotState::Free;
            self.slots[idx].submission = None;
            self.slots[idx].rx = None;
        }
        self.in_flight = None;
    }
}

/// Shared, lightweight WGPU headless rendering core used by both [`WgpuHeadlessPipeline`]
/// and [`WgpuScenePipeline`].
pub struct WgpuCore {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub target_texture: wgpu::Texture,
    pub target_view: wgpu::TextureView,
    ring: ReadbackRing,
    pub width: u32,
    pub height: u32,
    faulted: Arc<AtomicBool>,
    pub adapter_name: String,
    pub backend_name: String,
}

impl WgpuCore {
    /// Synchronously initializes the headless WGPU core.
    pub fn new(width: u32, height: u32, mode: RustWallpaperMode) -> Result<Self> {
        pollster::block_on(Self::new_async(width, height, mode))
    }

    /// Asynchronously initializes the WGPU instance, adapter, device, offscreen texture,
    /// and adaptive [`ReadbackRing`].
    pub async fn new_async(width: u32, height: u32, mode: RustWallpaperMode) -> Result<Self> {
        let backends = Self::resolve_backends();
        let flags = Self::resolve_instance_flags();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            flags,
            ..Default::default()
        });

        let preferred_power = Self::resolve_power_preference();
        let fallback_power = match preferred_power {
            wgpu::PowerPreference::LowPower => wgpu::PowerPreference::HighPerformance,
            _ => wgpu::PowerPreference::LowPower,
        };

        // Cascade adapter discovery: preferred power -> alternate power -> software fallback.
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: preferred_power,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) => a,
            None => match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: fallback_power,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
            {
                Some(a) => a,
                None => instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: true,
                    })
                    .await
                    .context("No compatible Vulkan or OpenGL ES adapter found for wgpu")?,
            },
        };

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        // Request downlevel defaults scaled to the adapter's max resolution for maximum
        // compatibility and low driver memory overhead across integrated and discrete GPUs.
        let required_limits =
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rwallpprs-wgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    ..Default::default()
                },
                None,
            )
            .await
            .context("Failed to open a logical wgpu::Device")?;

        let faulted = Arc::new(AtomicBool::new(false));
        let faulted_cl = Arc::clone(&faulted);
        device.on_uncaptured_error(Box::new(move |err: wgpu::Error| {
            eprintln!("[rwallpprs-sdk] Uncaptured WGPU error ({err}); marking device faulted");
            faulted_cl.store(true, Ordering::Relaxed);
        }));

        let safe_w = width.max(1);
        let safe_h = height.max(1);
        let (target_texture, target_view) = Self::create_render_targets(&device, safe_w, safe_h);

        let sync_override = std::env::var("RWALLPPRS_GPU_SYNC_READBACK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let async_pipelined = mode == RustWallpaperMode::Animated && !sync_override;

        let ring = ReadbackRing::new(&device, safe_w, safe_h, async_pipelined);

        Ok(Self {
            device,
            queue,
            target_texture,
            target_view,
            ring,
            width: safe_w,
            height: safe_h,
            faulted,
            adapter_name,
            backend_name,
        })
    }

    fn resolve_backends() -> wgpu::Backends {
        match std::env::var("RWALLPPRS_GPU_BACKEND")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "vulkan" | "vk" => wgpu::Backends::VULKAN,
            "gl" | "gles" | "opengl" => wgpu::Backends::GL,
            _ => wgpu::Backends::VULKAN | wgpu::Backends::GL,
        }
    }

    fn resolve_instance_flags() -> wgpu::InstanceFlags {
        let validation = std::env::var("RWALLPPRS_GPU_VALIDATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if validation {
            wgpu::InstanceFlags::default()
        } else {
            wgpu::InstanceFlags::empty()
        }
    }

    fn resolve_power_preference() -> wgpu::PowerPreference {
        match std::env::var("RWALLPPRS_GPU_POWER")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "high" | "performance" | "discrete" => wgpu::PowerPreference::HighPerformance,
            _ => wgpu::PowerPreference::LowPower,
        }
    }

    pub fn create_render_targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rwallpprs-offscreen-bgra"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Returns `true` if an uncaptured WGPU error occurred on this device.
    #[inline]
    pub fn has_faulted(&self) -> bool {
        self.faulted.load(Ordering::Relaxed)
    }

    /// Resizes the offscreen texture and staging buffers if the output resolution changed.
    pub fn resize_if_needed(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.width == w && self.height == h {
            return;
        }
        let (tex, view) = Self::create_render_targets(&self.device, w, h);
        self.target_texture = tex;
        self.target_view = view;
        self.ring.recreate(&self.device, w, h);
        self.width = w;
        self.height = h;
    }

    /// Copies the offscreen target texture into the active staging buffer, submits `encoder`,
    /// and transfers the completed frame into `canvas`.
    pub fn submit_and_readback(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        canvas: &mut Canvas<'_>,
    ) -> Result<()> {
        if self.has_faulted() {
            anyhow::bail!("wgpu device reported an uncaptured error");
        }

        let slot = self.ring.select_write_slot();

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: self.ring.slot_buffer(slot),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.ring.bytes_per_row()),
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
        self.ring
            .complete_frame(&self.device, canvas, slot, submission)?;

        if self.has_faulted() {
            anyhow::bail!("wgpu device faulted during frame submission");
        }
        Ok(())
    }

    /// Drains pending GPU submissions and unmaps all staging buffers while suspended.
    pub fn suspend(&mut self) {
        self.ring.release_all_mappings(&self.device);
    }
}

/// Layout of the uniform buffer (`@group(0) @binding(0) var<uniform> u: NiriShaderUniforms;`)
/// automatically bound for every [`GpuShaderWallpaper`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NiriShaderUniforms {
    /// Screen resolution in pixels `(width, height)`.
    pub resolution: [f32; 2],
    /// Seconds elapsed since wallpaper startup.
    pub time: f32,
    /// Frame delta time in seconds.
    pub dt: f32,
    /// Horizontal scroll offset of the Niri tiling strip.
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
    /// Initializes wallpaper state from CLI arguments.
    fn init(args: &[String]) -> Self;

    /// Returns the WGSL shader source.
    ///
    /// May provide only `@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>`
    /// (using the built-in fullscreen triangle vertex shader) or both `vs_main` and `fs_main`.
    fn wgsl_fragment_source(&self) -> &str;

    /// Preferred execution mode (`Animated`, `Static`, `ReactiveNiri`).
    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    /// Called when the daemon suspends rendering (e.g. fullscreen window occlusion).
    fn on_suspend(&mut self) {}

    /// Called when the daemon resumes rendering.
    fn on_resume(&mut self) {}

    /// CPU fallback used when no Vulkan/GL adapter is available.
    fn fallback_cpu_render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>);
}

/// Headless WGPU pipeline for fullscreen WGSL shader wallpapers.
pub struct WgpuHeadlessPipeline {
    core: WgpuCore,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl WgpuHeadlessPipeline {
    pub fn new(fragment_wgsl: &str, width: u32, height: u32) -> Result<Self> {
        Self::new_with_mode(fragment_wgsl, width, height, RustWallpaperMode::Animated)
    }

    pub fn new_with_mode(
        fragment_wgsl: &str,
        width: u32,
        height: u32,
        mode: RustWallpaperMode,
    ) -> Result<Self> {
        pollster::block_on(Self::new_async(fragment_wgsl, width, height, mode))
    }

    async fn new_async(
        fragment_wgsl: &str,
        width: u32,
        height: u32,
        mode: RustWallpaperMode,
    ) -> Result<Self> {
        let core = WgpuCore::new_async(width, height, mode).await?;

        // Only inject the default fullscreen triangle vertex shader if the user source
        // does not already define `fn vs_main`.
        let has_custom_vs = fragment_wgsl.contains("fn vs_main");
        let default_vs = if has_custom_vs {
            ""
        } else {
            r#"
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((in_vertex_index << 1u) & 2u);
    let y = f32(in_vertex_index & 2u);
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}
"#
        };

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
{default_vs}
{fragment_wgsl}
"#
        );

        let shader = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rwallpprs-wgsl-shader"),
                source: wgpu::ShaderSource::Wgsl(full_wgsl.into()),
            });

        let uniform_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("niri-uniform-buffer"),
            size: std::mem::size_of::<NiriShaderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            core.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("niri-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout =
            core.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rwallpprs-pipeline-layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = core
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        format: SCENE_TARGET_FORMAT,
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

        Ok(Self {
            core,
            pipeline,
            uniform_buffer,
            bind_group,
        })
    }

    pub fn adapter_summary(&self) -> String {
        format!("{} ({})", self.core.adapter_name, self.core.backend_name)
    }

    pub fn resize_if_needed(&mut self, width: u32, height: u32) {
        self.core.resize_if_needed(width, height);
    }

    pub fn suspend(&mut self) {
        self.core.suspend();
    }

    /// Executes the WGSL shader on the GPU and copies the finished frame into [`Canvas`].
    pub fn render_to_canvas(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> Result<()> {
        self.core.resize_if_needed(ctx.width, ctx.height);

        let uniforms = NiriShaderUniforms {
            resolution: [ctx.width as f32, ctx.height as f32],
            time: ctx.time_secs as f32,
            dt: ctx.dt_secs,
            niri_scroll_offset: ctx.niri.strip_scroll_offset as f32,
            niri_workspace_idx: ctx.niri.active_workspace_idx,
            niri_overview_open: if ctx.niri.overview_open { 1.0 } else { 0.0 },
            niri_windows_count: ctx.niri.visible_windows_count,
        };

        self.core
            .queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder =
            self.core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rwallpprs-shader-encoder"),
                });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rwallpprs-fullscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.core.target_view,
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

        self.core.submit_and_readback(encoder, canvas)
    }
}

/// Adapter turning any [`GpuShaderWallpaper`] into a standard [`Wallpaper`] with automatic
/// CPU fallback when Vulkan/GL is unavailable.
pub struct ShaderWallpaperAdapter<S: GpuShaderWallpaper> {
    inner: S,
    gpu: Option<WgpuHeadlessPipeline>,
    gpu_init_attempted: bool,
    ppm_dump_path: Option<String>,
}

impl<S: GpuShaderWallpaper> Wallpaper for ShaderWallpaperAdapter<S> {
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
        if let Some(gpu) = &mut self.gpu {
            gpu.suspend();
        }
        self.inner.on_suspend();
    }

    fn on_resume(&mut self) {
        self.inner.on_resume();
    }

    fn render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> bool {
        if !self.gpu_init_attempted {
            self.gpu_init_attempted = true;
            let mode = self.inner.default_mode();
            match WgpuHeadlessPipeline::new_with_mode(
                self.inner.wgsl_fragment_source(),
                ctx.width,
                ctx.height,
                mode,
            ) {
                Ok(pipeline) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU shader pipeline initialized on {} ({}x{}, mode={:?})",
                        pipeline.adapter_summary(),
                        ctx.width,
                        ctx.height,
                        mode
                    );
                    self.gpu = Some(pipeline);
                }
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU unavailable ({}), falling back to CPU renderer",
                        e
                    );
                }
            }
        }

        if let Some(gpu) = &mut self.gpu {
            match gpu.render_to_canvas(ctx, canvas) {
                Ok(()) => {
                    self.maybe_dump_ppm(ctx, canvas);
                    return true;
                }
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] Shader GPU pipeline error ({}), switching to CPU fallback",
                        e
                    );
                    self.gpu = None;
                }
            }
        }

        self.inner.fallback_cpu_render(ctx, canvas);
        self.maybe_dump_ppm(ctx, canvas);
        true
    }
}

impl<S: GpuShaderWallpaper> ShaderWallpaperAdapter<S> {
    fn maybe_dump_ppm(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) {
        if let Some(path) = self.ppm_dump_path.take() {
            if let Err(e) = write_ppm_dump(&path, ctx.width, ctx.height, canvas.pixels_mut()) {
                eprintln!("[rwallpprs-sdk] Failed to write PPM dump '{}': {}", path, e);
            } else {
                self.ppm_dump_path = Some(path);
            }
        }
    }
}

/// Per-frame GPU context handed to [`GpuSceneWallpaper::render_scene`].
///
/// Gives full access to [`wgpu::Device`], [`wgpu::Queue`], and [`wgpu::CommandEncoder`]
/// for custom pipelines, vertex/index buffers, instancing, and multi-pass rendering.
pub struct GpuScene<'a> {
    /// Logical `wgpu` device for creating buffers, pipelines, bind groups, and textures.
    pub device: &'a wgpu::Device,
    /// `wgpu` queue for uploading uniform/vertex/instance data (`write_buffer`, `write_texture`).
    pub queue: &'a wgpu::Queue,
    /// Command encoder for the current frame.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Target offscreen texture view (`Bgra8Unorm`) for the current frame.
    pub target: &'a wgpu::TextureView,
    /// Target texture format ([`SCENE_TARGET_FORMAT`] = `Bgra8Unorm`).
    pub format: wgpu::TextureFormat,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// High-level trait for custom multi-pipeline / instanced `wgpu` scene wallpapers.
pub trait GpuSceneWallpaper: Sized + 'static {
    /// Initializes wallpaper state from CLI arguments.
    fn init(args: &[String]) -> Self;

    /// Preferred execution mode (`Animated`, `Static`, `ReactiveNiri`).
    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    /// Called when the daemon suspends rendering (e.g. fullscreen window occlusion).
    fn on_suspend(&mut self) {}

    /// Called when the daemon resumes rendering.
    fn on_resume(&mut self) {}

    /// Encodes one scene frame into `scene.encoder`, rendering into `scene.target`.
    ///
    /// Returns `true` when a new frame was rendered. Returning `false` skips the GPU
    /// readback copy and tells the daemon to reuse the previous frame.
    fn render_scene(&mut self, ctx: &FrameContext, scene: &mut GpuScene<'_>) -> bool;

    /// CPU fallback used when no Vulkan/GL adapter is available.
    fn fallback_cpu_render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>);
}

/// Headless scene pipeline wrapping [`WgpuCore`].
pub struct WgpuScenePipeline {
    core: WgpuCore,
}

impl WgpuScenePipeline {
    pub async fn new_async(width: u32, height: u32, mode: RustWallpaperMode) -> Result<Self> {
        let core = WgpuCore::new_async(width, height, mode).await?;
        Ok(Self { core })
    }

    pub fn adapter_summary(&self) -> String {
        format!("{} ({})", self.core.adapter_name, self.core.backend_name)
    }

    pub fn suspend(&mut self) {
        self.core.suspend();
    }

    /// Invokes `inner.render_scene` and, if the scene rendered a frame (`changed == true`),
    /// copies the offscreen target into `canvas` via the adaptive [`ReadbackRing`].
    fn render_frame<S: GpuSceneWallpaper>(
        &mut self,
        inner: &mut S,
        ctx: &FrameContext,
        canvas: &mut Canvas<'_>,
    ) -> Result<bool> {
        self.core.resize_if_needed(ctx.width, ctx.height);

        let mut encoder =
            self.core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rwallpprs-scene-encoder"),
                });

        let changed = {
            let mut scene = GpuScene {
                device: &self.core.device,
                queue: &self.core.queue,
                encoder: &mut encoder,
                target: &self.core.target_view,
                format: SCENE_TARGET_FORMAT,
                width: self.core.width,
                height: self.core.height,
            };
            inner.render_scene(ctx, &mut scene)
        };

        // Do not waste GPU bandwidth or overwrite staging slots when the wallpaper skipped the frame.
        if !changed {
            return Ok(false);
        }

        self.core.submit_and_readback(encoder, canvas)?;
        Ok(true)
    }
}

/// Writes a [`Canvas`] frame (`ARGB8888` little-endian) to a binary `P6` PPM file.
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

/// Adapter turning any [`GpuSceneWallpaper`] into a standard [`Wallpaper`] with lazy GPU
/// initialization, suspend resource cleanup, and automatic CPU fallback.
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
        if let Some(gpu) = &mut self.gpu {
            gpu.suspend();
        }
        self.inner.on_suspend();
    }

    fn on_resume(&mut self) {
        self.inner.on_resume();
    }

    fn render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> bool {
        if !self.gpu_init_attempted {
            self.gpu_init_attempted = true;
            let mode = self.inner.default_mode();
            match pollster::block_on(WgpuScenePipeline::new_async(ctx.width, ctx.height, mode)) {
                Ok(gpu) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU scene pipeline initialized on {} ({}x{}, mode={:?})",
                        gpu.adapter_summary(),
                        ctx.width,
                        ctx.height,
                        mode
                    );
                    self.gpu = Some(gpu);
                }
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU unavailable ({}), using CPU fallback renderer",
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
                        "[rwallpprs-sdk] Scene GPU pipeline error ({}), switching to CPU fallback",
                        e
                    );
                    self.gpu = None;
                }
            }
        }

        if self.gpu.is_none() {
            self.inner.fallback_cpu_render(ctx, canvas);
            changed = true;
        }

        if changed {
            if let Some(path) = self.ppm_dump_path.take() {
                if let Err(e) = write_ppm_dump(&path, ctx.width, ctx.height, canvas.pixels_mut()) {
                    eprintln!("[rwallpprs-sdk] Failed to write PPM dump '{}': {}", path, e);
                } else {
                    self.ppm_dump_path = Some(path);
                }
            }
        }

        changed
    }
}

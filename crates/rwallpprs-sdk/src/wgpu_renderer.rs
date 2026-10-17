//! Аппаратный GPU-рендерер для Rust-обоев на базе [`wgpu`] (Vulkan / GL / Metal).
//!
//! Позволяет писать фрагментные шейдеры на языке **WGSL**, которые исполняются на видеокарте
//! в безголовом режиме (Offscreen Texture `Bgra8Unorm`), автоматически получают состояние
//! композитора **Niri** через Uniform-буфер [`NiriShaderUniforms`] и выгружают готовые пиксели
//! (`BGRA8` / `ARGB8888`) напрямую в разделяемую память `memfd` демона!

use crate::{Canvas, FrameContext, RustWallpaperMode, Wallpaper};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::mpsc;

/// Структура Uniform-буфера (`@group(0) @binding(0) var<uniform> u: NiriShaderUniforms;`),
/// доступная внутри любого WGSL-шейдера обоев.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NiriShaderUniforms {
    /// Разрешение экрана в пикселях `(width, height)`.
    pub resolution: [f32; 2],
    /// Время с момента старта в секундах.
    pub time: f32,
    /// Дельта времени кадра в секундах.
    pub dt: f32,
    /// Горизонтальное смещение скролла ленты Niri.
    pub niri_scroll_offset: f32,
    /// Индекс активного воркспейса Niri.
    pub niri_workspace_idx: u32,
    /// `1.0`, если открыт режим Niri Overview (`Mod+O`), иначе `0.0`.
    pub niri_overview_open: f32,
    /// Количество видимых окон на активном воркспейсе Niri.
    pub niri_windows_count: u32,
}

/// Высокоуровневый типаж для создания аппаратных GPU-обоев на базе WGSL-шейдера.
pub trait GpuShaderWallpaper: Sized + 'static {
    /// Инициализация параметров обоев.
    fn init(args: &[String]) -> Self;

    /// Исходный код шейдера на языке WGSL (должен содержать `vs_main` и `fs_main`,
    /// либо только `fs_main(in: VertexOutput) -> @location(0) vec4<f32>` при использовании
    /// стандартного полноэкранного треугольника).
    fn wgsl_fragment_source(&self) -> &str;

    /// Режим работы (`Animated`, `Static`, `ReactiveNiri`).
    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    /// Программный CPU-фоллбэк на случай запуска в окружении без GPU/Vulkan.
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
    staging_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
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
            .context("Не найден совместимый адаптер Vulkan/GL для wgpu")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .context("Не удалось открыть логическое устройство wgpu::Device")?;

        // Собираем полный шейдерный модуль со встроенным вершинным шейдером полноэкранного треугольника
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

        let (target_texture, target_view, staging_buffer, padded_bytes_per_row) =
            Self::create_render_targets(&device, width, height);

        Ok(Self {
            device,
            queue,
            pipeline,
            uniform_buffer,
            bind_group,
            target_texture,
            target_view,
            staging_buffer,
            width,
            height,
            padded_bytes_per_row,
        })
    }

    fn create_render_targets(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Buffer, u32) {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded = width * 4;
        let padded_bytes_per_row = unpadded.div_ceil(align) * align;

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

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rwallpprs-staging-readback"),
            size: (padded_bytes_per_row as u64) * (height as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        (texture, view, staging_buffer, padded_bytes_per_row)
    }

    pub fn resize_if_needed(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let (tex, view, buf, stride) = Self::create_render_targets(&self.device, width, height);
        self.target_texture = tex;
        self.target_view = view;
        self.staging_buffer = buf;
        self.padded_bytes_per_row = stride;
        self.width = width;
        self.height = height;
    }

    /// Исполняет WGSL-шейдер на GPU и копирует результат напрямую в zero-copy `Canvas` (`memfd`).
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
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        {
            let mapped = slice.get_mapped_range();
            let dest_bytes: &mut [u8] = bytemuck::cast_slice_mut(canvas.pixels_mut());
            let row_bytes = (self.width as usize) * 4;
            let padded_stride = self.padded_bytes_per_row as usize;

            if padded_stride == row_bytes {
                let total = row_bytes * (self.height as usize);
                dest_bytes[..total].copy_from_slice(&mapped[..total]);
            } else {
                for y in 0..(self.height as usize) {
                    let src_off = y * padded_stride;
                    let dst_off = y * row_bytes;
                    dest_bytes[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&mapped[src_off..src_off + row_bytes]);
                }
            }
        }
        self.staging_buffer.unmap();
        Ok(())
    }
}

/// Адаптер, превращающий любой [`GpuShaderWallpaper`] в стандартный [`Wallpaper`] с автоматическим
/// фоллбэком на CPU, если Vulkan/GL недоступен.
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
                        "[rwallpprs-sdk] WGPU (Vulkan/GL) шейдерный конвейер успешно инициализирован ({}x{})",
                        ctx.width, ctx.height
                    );
                    self.gpu = Some(pipeline);
                }
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU недоступен ({}), используется CPU-рендерер",
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

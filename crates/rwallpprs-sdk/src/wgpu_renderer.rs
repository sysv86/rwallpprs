//! Аппаратный GPU-рендерер для Rust-обоев на базе [`wgpu`] (Vulkan / GL / Metal).
//!
//! Позволяет писать фрагментные шейдеры на языке **WGSL**, которые исполняются на видеокарте
//! в безголовом режиме (Offscreen Texture `Bgra8Unorm`), автоматически получают состояние
//! композитора **Niri** через Uniform-буфер [`NiriShaderUniforms`] и выгружают готовые пиксели
//! (`BGRA8` / `ARGB8888`) напрямую в разделяемую память `memfd` демона!
//!
//! Для сложных 2D-сцен (тысячи линий/кругов/полигонов, инстансинг, собственные пайплайны
//! и несколько проходов) используйте сценовый API: типаж [`GpuSceneWallpaper`], контекст кадра
//! [`GpuScene`] и адаптер [`SceneWallpaperAdapter`] с макросом `run_scene_wallpaper!`.

use crate::{Canvas, FrameContext, RustWallpaperMode, Wallpaper};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::mpsc::TryRecvError;

/// Формат offscreen-текстуры (`Bgra8Unorm`), в которую рисуют сценовые обои.
///
/// Совпадает с порядком байт Wayland `ARGB8888` / `XRGB8888` на Little-Endian,
/// поэтому готовый кадр можно memcpy'ить прямо в [`Canvas`] без конвертации.
pub const SCENE_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// Распаковывает построчно отображённый staging-буфер (с выравниванием строки по 256 байт)
/// в пиксели [`Canvas`], игнорируя padding каждой строки.
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

/// Количество staging-буферов в пуле readback'а.
///
/// Двух хватает для конвейера «пока GPU рисует кадр `N`, CPU читает кадр `N-1`»,
/// третий даёт запас, чтобы CPU не упирался в занятость буферов при коротких
/// всплесках задержки GPU.
const READBACK_SLOTS: usize = 3;

/// Состояние одного staging-буфера пула readback'а.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// Буфер свободен (не замаплен), в него можно писать копию нового кадра.
    Free,
    /// Копия отправлена на GPU, `map_async` ждёт завершения сабмита.
    InFlight,
    /// `map_async` завершился успешно — данные готовы к копированию в [`Canvas`].
    Mapped,
}

/// Один staging-буфер пула вместе с состоянием его readback'а.
struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: SlotState,
    /// Сабмит, в котором буфер был заполнен (для точечного ожидания `WaitForSubmissionIndex`).
    submission: Option<wgpu::SubmissionIndex>,
    /// Приёмник результата `map_async` текущего использования слота.
    rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// Пул staging-буферов с двойной (тройной) буферизацией GPU→CPU readback'а.
///
/// Вместо синхронной пары «`submit` → `Maintain::Wait` → `unmap`» на каждый кадр пул
/// держит несколько буферов и слот состояния для каждого. На кадре `N` копия текстуры
/// пишется в свободный слот, а в [`Canvas`] выдаётся уже посчитанный кадр `N-1`: пока
/// GPU занят кадром `N`, CPU спокойно забирает `N-1`. За счёт этого
/// `device.poll(Maintain::Wait)` больше не вызывается на только что отправленный сабмит.
///
/// Цена — задержка вывода в один кадр (для обоев несущественно). Первый кадр и кадр
/// после смены размера выдаются синхронно, чтобы [`Canvas`] никогда не оставался пустым.
struct StagingPool {
    slots: Vec<ReadbackSlot>,
    /// Индексы слотов в порядке отправки — очередь на выдачу в [`Canvas`].
    pending: VecDeque<usize>,
    /// Кэш выбора свободного слота (round-robin).
    cursor: usize,
    /// Выровненная (по 256 байт) длина строки staging-буфера.
    bytes_per_row: u32,
    /// Сколько кадров уже выдано в [`Canvas`] (нужно для синхронного первого кадра).
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

    /// Пересобирает пул под новый размер кадра и сбрасывает конвейер: ближайший кадр
    /// снова будет выдан синхронно (см. [`Self::finish_frame`]).
    fn recreate(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Замапленные буферы сначала аккуратно снимаем с маппинга, затем отпускаем.
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

    /// Выровненная длина строки, которую нужно указывать в `copy_texture_to_buffer`.
    fn bytes_per_row(&self) -> u32 {
        self.bytes_per_row
    }

    /// Возвращает буфер слота, в который нужно записать копию текстуры кадра.
    fn slot_buffer(&self, idx: usize) -> &wgpu::Buffer {
        &self.slots[idx].buffer
    }

    /// Подыскивает свободный слот под копию нового кадра.
    ///
    /// Если все слоты заняты (GPU хронически отстаёт), принудительно выдаёт самый
    /// старый кадр — блокируясь на его сабмите, а не на только что отправленном.
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

    /// Регистрирует `map_async` для только что отправленного кадра и выдаёт в [`Canvas`]
    /// уже посчитанный более старый кадр (в установившемся конвейере — `N-1`).
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

        // Первый кадр (и кадр после resize) выдаём синхронно, чтобы `Canvas` не остался
        // пустым. Дальше держим конвейер глубиной в один кадр: выдаём `N-1`, не трогая
        // свежий `N` — именно это даёт перекрытие работы GPU и CPU.
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
        // Сюда попадать не должны: `begin_frame` освобождает слот при заполнении пула.
        self.cursor = (self.cursor + 1) % self.slots.len();
        self.cursor
    }

    /// Выдаёт самый старый кадр из очереди в [`Canvas`] и освобождает его слот.
    fn deliver_oldest(&mut self, device: &wgpu::Device, canvas: &mut Canvas<'_>) -> Result<()> {
        let idx = match self.pending.front() {
            Some(&idx) => idx,
            None => return Ok(()),
        };

        // Сначала дешёвый неблокирующий опрос: обычно кадр `N-1` уже готов.
        let mut ready = self.poll_slot(idx)?;
        if !ready {
            // Ждём именно сабмит этого кадра (`N-1`), а не самый свежий (`N`): так GPU
            // продолжает работать над `N`, пока CPU забирает `N-1`.
            match self.slots[idx].submission.take() {
                Some(sub) => device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub)),
                None => device.poll(wgpu::Maintain::Wait),
            };
            ready = self.poll_slot(idx)?;
            if !ready {
                // Перестраховка для бэкендов без поддержки точечного ожидания.
                device.poll(wgpu::Maintain::Wait);
                ready = self.poll_slot(idx)?;
            }
        }

        if !ready {
            anyhow::bail!("staging-буфер readback'а так и не замапился");
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

    /// Неблокирующая проверка: завершился ли `map_async` слота.
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
                anyhow::bail!("map_async staging-буфера завершился ошибкой: {e}")
            }
            Some(Err(TryRecvError::Empty)) => Ok(false),
            Some(Err(TryRecvError::Disconnected)) => {
                anyhow::bail!("callback map_async staging-буфера потерян")
            }
            None => Ok(false),
        }
    }
}

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

        // Слот-приёмник копии этого кадра; попутно пул может выдать в canvas кадр N-1.
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

/// Контекст одного кадра для **сценовых** GPU-обоев.
///
/// В отличие от [`GpuShaderWallpaper`] (один фрагментный WGSL-шейдер), сценовые обои получают
/// полноценный доступ к [`wgpu::Device`], [`wgpu::Queue`] и [`wgpu::CommandEncoder`], поэтому могут
/// создавать собственные пайплайны, вершинные/индексные буферы, инстансинг (тысячи линий, кругов,
/// полигонов), несколько render pass'ов и промежуточные текстуры.
///
/// Обои САМИ записывают свои render pass'ы в [`GpuScene::encoder`], рисуя в [`GpuScene::target`]
/// (формат [`GpuScene::format`] = [`SCENE_TARGET_FORMAT`]). SDK сам выгружает результат в [`Canvas`].
pub struct GpuScene<'a> {
    /// Логическое устройство `wgpu` (создание буферов, пайплайнов, текстур и т.д.).
    pub device: &'a wgpu::Device,
    /// Очередь `wgpu` (загрузка данных в буферы/текстуры через `write_buffer` / `write_texture`).
    pub queue: &'a wgpu::Queue,
    /// Кодировщик команд текущего кадра — сюда обои пишут свои render pass'ы.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Целевая offscreen-текстура кадра (`Bgra8Unorm`), в которую нужно рисовать.
    pub target: &'a wgpu::TextureView,
    /// Формат целевой текстуры (`Bgra8Unorm`).
    pub format: wgpu::TextureFormat,
    /// Ширина кадра в пикселях.
    pub width: u32,
    /// Высота кадра в пикселях.
    pub height: u32,
}

/// Высокоуровневый типаж для **сценовых** аппаратных обоев на базе полноценного `wgpu`.
///
/// Используйте его, когда фрагментного шейдера недостаточно: тысячи инстансов геометрии,
/// собственные пайплайны, несколько проходов, вычислительные шейдеры подготовки данных и т.п.
pub trait GpuSceneWallpaper: Sized + 'static {
    /// Инициализация параметров обоев.
    fn init(args: &[String]) -> Self;

    /// Режим работы (`Animated`, `Static`, `ReactiveNiri`).
    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    /// Вызывается демоном при засыпании/скрытии обоев (например, для паузы симуляции).
    fn on_suspend(&mut self) {}

    /// Вызывается демоном при возобновлении обоев.
    fn on_resume(&mut self) {}

    /// Рисует один кадр сцены, записывая render pass'ы в `scene.encoder` и рисуя в `scene.target`.
    ///
    /// Возвращает `true`, если кадр изменился (демон обновит стол). `false` допустим для пауз,
    /// когда содержимое не поменялось — демон пропустит кадр без перерисовки.
    fn render_scene(&mut self, ctx: &FrameContext, scene: &mut GpuScene<'_>) -> bool;

    /// Программный CPU-фоллбэк на случай запуска в окружении без GPU/Vulkan.
    ///
    /// Обязателен: минимальная картинка, чтобы обои жили без GPU.
    fn fallback_cpu_render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>);
}

/// Внутренний headless-конвейер сценовых обоев: offscreen-текстура + пул staging-буферов
/// для асинхронного readback'а.
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
            .context("Не найден совместимый адаптер Vulkan/GL для wgpu")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .context("Не удалось открыть логическое устройство wgpu::Device")?;

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

    /// Даёт обоям отрисовать сцену, затем копирует offscreen-текстуру в [`Canvas`] (`memfd`).
    ///
    /// Копия кадра `N` отправляется в свободный слот пула и НЕ дожидается на месте:
    /// за кадр выдаётся уже посчитанный кадр `N-1` (`StagingPool::deliver_oldest`).
    fn render_frame<S: GpuSceneWallpaper>(
        &mut self,
        inner: &mut S,
        ctx: &FrameContext,
        canvas: &mut Canvas<'_>,
    ) -> Result<bool> {
        self.resize_if_needed(ctx.width, ctx.height);

        // Слот-приёмник копии этого кадра; попутно пул может выдать в canvas кадр N-1.
        let slot = self.pool.begin_frame(&self.device, canvas)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rwallpprs-scene-encoder"),
            });

        // Отдаём обои собственный encoder и target; они пишут туда свои render pass'ы.
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

/// Записывает текущий кадр в PPM (`P6`) для визуальной проверки без запуска демона.
///
/// Путь берётся из env-переменной `RWALLPPRS_DUMP_PPM` (см. [`SceneWallpaperAdapter`]).
/// Пиксели [`Canvas`] хранятся в `BGRA` (Little-Endian `ARGB8888`), в PPM кладём `RGB` без альфы.
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

/// Адаптер, превращающий любой [`GpuSceneWallpaper`] в стандартный [`Wallpaper`] с ленивой
/// инициализацией GPU и автоматическим фоллбэком на CPU, если Vulkan/GL недоступен.
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
                        "[rwallpprs-sdk] WGPU (Vulkan/GL) сценовый конвейер успешно инициализирован ({}x{})",
                        ctx.width, ctx.height
                    );
                    self.gpu = Some(gpu);
                }
                Err(e) => {
                    eprintln!(
                        "[rwallpprs-sdk] WGPU недоступен ({}), используется CPU-рендерер",
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
                        "[rwallpprs-sdk] Ошибка сценового GPU-конвейера ({}), переход на CPU-рендерер",
                        e
                    );
                    self.gpu = None;
                }
            }
        }

        // GPU не инициализировался или отвалился — рисуем минимальную CPU-картинку.
        if self.gpu.is_none() {
            self.inner.fallback_cpu_render(ctx, canvas);
            changed = true;
        }

        if let Some(path) = self.ppm_dump_path.take() {
            if let Err(e) = write_ppm_dump(&path, ctx.width, ctx.height, canvas.pixels_mut()) {
                eprintln!("[rwallpprs-sdk] Не удалось записать PPM-дамп '{}': {}", path, e);
            } else {
                self.ppm_dump_path = Some(path);
            }
        }

        changed
    }
}

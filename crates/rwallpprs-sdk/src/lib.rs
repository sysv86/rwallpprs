//! # RWallpprs SDK (`v0.2.0-beta.1`)
//!
//! Специальное API для написания обоев под композитор **Niri** на чистом **Rust**:
//! - **На GPU (`wgpu` + WGSL шейдеры)**: через типаж [`GpuShaderWallpaper`] и макрос [`run_shader_wallpaper!`].
//! - **На CPU (`Canvas` zero-copy SHM)**: через типаж [`Wallpaper`] и макрос [`run_wallpaper!`].
//! - Все операции с разделяемой памятью (`RWALLPPRS_SHM_FD`) используют безопасные обертки [`rustix::mm::mmap`].

pub mod wgpu_renderer;

pub use rwallpprs_ipc::{NiriViewportState, RustWallpaperMode, WorkerAck, WorkerEvent};
pub use wgpu_renderer::{
    GpuShaderWallpaper, NiriShaderUniforms, ShaderWallpaperAdapter, WgpuHeadlessPipeline,
};

use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};
use std::io::{BufRead, BufReader, Write};
use std::os::fd::{BorrowedFd, RawFd};
use std::ptr::NonNull;

/// Цвет пикселя в формате Wayland `ARGB8888` / `XRGB8888` (Little-Endian: `[B, G, R, A]`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Color {
    pub b: u8,
    pub g: u8,
    pub r: u8,
    pub a: u8,
}

impl Color {
    #[inline(always)]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { b, g, r, a: 255 }
    }

    #[inline(always)]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { b, g, r, a }
    }

    #[inline(always)]
    pub const fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as u8,
            g: ((hex >> 8) & 0xFF) as u8,
            b: (hex & 0xFF) as u8,
            a: 255,
        }
    }

    #[inline(always)]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        Self {
            r: (self.r as f32 * inv + other.r as f32 * t) as u8,
            g: (self.g as f32 * inv + other.g as f32 * t) as u8,
            b: (self.b as f32 * inv + other.b as f32 * t) as u8,
            a: (self.a as f32 * inv + other.a as f32 * t) as u8,
        }
    }

    #[inline(always)]
    pub const fn to_u32_argb(self) -> u32 {
        ((self.a as u32) << 24)
            | ((self.r as u32) << 16)
            | ((self.g as u32) << 8)
            | (self.b as u32)
    }
}

/// Контекст кадра, передаваемый в метод `Wallpaper::render`.
#[derive(Debug, Clone)]
pub struct FrameContext {
    pub frame_index: u64,
    pub time_secs: f64,
    pub dt_secs: f32,
    pub width: u32,
    pub height: u32,
    pub niri: NiriViewportState,
}

/// Быстрый холст (`Canvas`), смотрящий напрямую в маппированный участок разделяемой памяти Wayland (`wl_shm`).
pub struct Canvas<'a> {
    pub width: u32,
    pub height: u32,
    pixels: &'a mut [u32],
}

impl<'a> Canvas<'a> {
    pub fn from_raw_slice(width: u32, height: u32, pixels: &'a mut [u32]) -> Self {
        assert!(pixels.len() >= (width as usize) * (height as usize));
        Self {
            width,
            height,
            pixels,
        }
    }

    #[inline(always)]
    pub fn pixels_mut(&mut self) -> &mut [u32] {
        self.pixels
    }

    #[inline]
    pub fn clear(&mut self, color: Color) {
        self.pixels.fill(color.to_u32_argb());
    }

    #[inline(always)]
    pub fn set_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            self.pixels[idx] = color.to_u32_argb();
        }
    }

    pub fn fill_vertical_gradient(&mut self, top: Color, bottom: Color) {
        let w = self.width as usize;
        let h = self.height as usize;
        for y in 0..h {
            let t = y as f32 / (h.max(1) as f32);
            let row_val = top.lerp(bottom, t).to_u32_argb();
            let row_start = y * w;
            self.pixels[row_start..row_start + w].fill(row_val);
        }
    }

    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Color) {
        let mut x0 = x0;
        let mut y0 = y0;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            self.set_pixel(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    pub fn fill_circle_sdf(&mut self, cx: f32, cy: f32, radius: f32, color: Color) {
        let min_x = ((cx - radius - 1.0).floor() as i32).max(0);
        let max_x = ((cx + radius + 1.0).ceil() as i32).min(self.width as i32 - 1);
        let min_y = ((cy - radius - 1.0).floor() as i32).max(0);
        let max_y = ((cy + radius + 1.0).ceil() as i32).min(self.height as i32 - 1);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= radius {
                    self.set_pixel(x, y, color);
                }
            }
        }
    }

    pub fn shade_uv<F>(&mut self, mut shader: F)
    where
        F: FnMut(f32, f32) -> Color,
    {
        let w = self.width as usize;
        let h = self.height as usize;
        let inv_w = 1.0 / (w.max(1) as f32);
        let inv_h = 1.0 / (h.max(1) as f32);

        for y in 0..h {
            let v = (y as f32) * inv_h;
            let row = &mut self.pixels[y * w..(y + 1) * w];
            for (x, px) in row.iter_mut().enumerate() {
                let u = (x as f32) * inv_w;
                *px = shader(u, v).to_u32_argb();
            }
        }
    }
}

/// Главный типаж для CPU/гибридных обоев на Rust.
pub trait Wallpaper: Sized + 'static {
    fn init(args: &[String]) -> Self;

    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    fn on_suspend(&mut self) {}
    fn on_resume(&mut self) {}

    fn render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> bool;
}

/// RAII-обертка над `rustix::mm::mmap`, гарантирующая автоматический `munmap` при выходе.
struct MappedShmRegion {
    ptr: NonNull<u8>,
    size: usize,
}

impl MappedShmRegion {
    fn map_fd(raw_fd: RawFd, size: usize) -> anyhow::Result<Self> {
        let borrowed = unsafe { BorrowedFd::borrow_raw(raw_fd) };
        let raw_ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                borrowed,
                0,
            )?
        };
        let ptr = NonNull::new(raw_ptr.cast::<u8>())
            .ok_or_else(|| anyhow::anyhow!("mmap вернул нулевой указатель"))?;
        Ok(Self { ptr, size })
    }

    unsafe fn slot_slice_mut(&mut self, offset: usize, pixel_count: usize) -> &mut [u32] {
        let slot_ptr = self.ptr.as_ptr().add(offset).cast::<u32>();
        std::slice::from_raw_parts_mut(slot_ptr, pixel_count)
    }
}

impl Drop for MappedShmRegion {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr.as_ptr().cast(), self.size);
        }
    }
}

/// Запускает цикл жизни Rust-обоев.
pub fn run_wallpaper_loop<W: Wallpaper>() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut wallpaper = W::init(&args);

    let shm_fd_str = match std::env::var("RWALLPPRS_SHM_FD") {
        Ok(val) => val,
        Err(_) => {
            eprintln!(
                "[rwallpprs-sdk] Автономная проверка (mode = {:?}). Для установки на стол Niri: rwallpprs set-bin <ПУТЬ_К_БИНАРНИКУ>",
                wallpaper.default_mode()
            );
            let width = 1920u32;
            let height = 1080u32;
            let mut buffer = vec![0u32; (width * height) as usize];
            let mut canvas = Canvas::from_raw_slice(width, height, &mut buffer);
            let ctx = FrameContext {
                frame_index: 0,
                time_secs: 0.0,
                dt_secs: 1.0 / 60.0,
                width,
                height,
                niri: NiriViewportState::default(),
            };
            let rendered = wallpaper.render(&ctx, &mut canvas);
            eprintln!(
                "[rwallpprs-sdk] Кадр {}x{} успешно отрендерен (rendered={}).",
                width, height, rendered
            );
            return Ok(());
        }
    };

    let shm_fd: RawFd = shm_fd_str.parse()?;
    let shm_size: usize = std::env::var("RWALLPPRS_SHM_SIZE")?.parse()?;
    let mut shm_region = MappedShmRegion::map_fd(shm_fd, shm_size)?;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    let reader = BufReader::new(stdin.lock());

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event: WorkerEvent = serde_json::from_str(&line)?;
        match event {
            WorkerEvent::RenderFrame {
                slot,
                frame_index,
                time_secs,
                dt_secs,
                width,
                height,
                niri,
            } => {
                let header = rwallpprs_ipc::ShmBufferHeader::new(width, height);
                let offset = header.slot_offset(slot);
                let pixel_count = (width as usize) * (height as usize);

                let slice = unsafe { shm_region.slot_slice_mut(offset, pixel_count) };
                let mut canvas = Canvas::from_raw_slice(width, height, slice);
                let ctx = FrameContext {
                    frame_index,
                    time_secs,
                    dt_secs,
                    width,
                    height,
                    niri,
                };

                let changed = wallpaper.render(&ctx, &mut canvas);
                let ack = if changed {
                    WorkerAck::FrameReady {
                        slot,
                        frame_index,
                        damaged_rect: Some((0, 0, width, height)),
                    }
                } else {
                    WorkerAck::FrameSkipped { frame_index }
                };

                serde_json::to_writer(&mut stdout, &ack)?;
                stdout.write_all(b"
")?;
                stdout.flush()?;
            }
            WorkerEvent::Suspend => wallpaper.on_suspend(),
            WorkerEvent::Resume => wallpaper.on_resume(),
            WorkerEvent::Terminate => break,
        }
    }

    Ok(())
}

/// Макрос для запуска CPU/гибридных обоев, реализующих [`Wallpaper`].
#[macro_export]
macro_rules! run_wallpaper {
    ($wallpaper_type:ty) => {
        fn main() -> anyhow::Result<()> {
            $crate::run_wallpaper_loop::<$wallpaper_type>()
        }
    };
}

/// Макрос для запуска аппаратных WGPU/WGSL шейдерных обоев, реализующих [`GpuShaderWallpaper`].
#[macro_export]
macro_rules! run_shader_wallpaper {
    ($shader_wallpaper_type:ty) => {
        fn main() -> anyhow::Result<()> {
            $crate::run_wallpaper_loop::<$crate::ShaderWallpaperAdapter<$shader_wallpaper_type>>()
        }
    };
}

//! Нативный аппаратный декодер видео на базе крейтов [`ffmpeg_next`] и [`ffmpeg_sys_next`].
//!
//! Полностью заменяет внешний CLI-процесс `ffmpeg`:
//! - Демультиплексирование контейнеров (`.mp4`, `.mkv`, `.webm`) через `ffmpeg_next::format::input`.
//! - Поддержка кодеков **H.264**, **HEVC (H.265)**, **VP9** и **AV1**.
//! - Инициализация аппаратного контекста `AVHWDeviceType` (`VA-API` на `/dev/dri/renderD128`,
//!   `Vulkan`, `CUDA/NVDEC`, `VDPAU`) через `av_hwdevice_ctx_create` с автоматическим фоллбэком
//!   на многопоточный программный декодер (`libdav1d` / `ff_h264`).
//! - Выгрузка аппаратных поверхностей через `av_hwframe_transfer_data` и цветовое преобразование +
//!   масштабирование через `ffmpeg_next::software::scaling::Context` (`libswscale`) напрямую
//!   в срез `BGRA` (`wl_shm::Format::Xrgb8888`) без промежуточных буферов.

use anyhow::{Context as AnyhowContext, Result};
use ffmpeg_next::{
    codec::{self, Id as CodecId},
    ffi::{
        av_buffer_ref, av_buffer_unref, av_hwdevice_ctx_create, av_hwframe_transfer_data,
        AVBufferRef, AVHWDeviceType, AVPixelFormat,
    },
    format::{self, Pixel},
    media::Type as MediaType,
    software::scaling::{context::Context as SwsContext, flag::Flags as SwsFlags},
    util::frame::video::Video as VideoFrame,
};
use rwallpprs_ipc::{HwAccelBackend, ScaleMode, VideoCodec};
use std::ffi::CString;
use std::path::PathBuf;
use std::ptr;
use tracing::{info, warn};

pub struct VideoWallpaperEngine {
    pub path: PathBuf,
    pub codec: VideoCodec,
    pub hw_accel: HwAccelBackend,
    pub scale_mode: ScaleMode,
    pub target_fps: u32,
    pub width: u32,
    pub height: u32,
    pub paused: bool,
    ictx: format::context::Input,
    video_stream_index: usize,
    decoder: codec::decoder::Video,
    hw_device_ctx: *mut AVBufferRef,
    sws_scaler: Option<SwsContext>,
    last_src_format: Option<(Pixel, u32, u32)>,
    decoded_frame: VideoFrame,
    sw_transfer_frame: VideoFrame,
    scaled_bgra_frame: VideoFrame,
    /// Уже отправляли decoder.send_eof() — drain проделан, вход исчерпан.
    eof_flushed: bool,
}

unsafe impl Send for VideoWallpaperEngine {}

impl VideoWallpaperEngine {
    pub fn new(
        path: PathBuf,
        hw_accel: HwAccelBackend,
        scale_mode: ScaleMode,
        target_fps: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        ffmpeg_next::init().context("Не удалось инициализировать библиотеки FFmpeg (ffmpeg-next)")?;

        let ictx = format::input(&path)
            .with_context(|| format!("Не удалось открыть видеоконтейнер {:?}", path))?;

        let stream = ictx
            .streams()
            .best(MediaType::Video)
            .ok_or_else(|| anyhow::anyhow!("В контейнере {:?} не найден видеопоток", path))?;

        let video_stream_index = stream.index();
        let codec_params = stream.parameters();
        let codec_id = codec_params.id();

        let detected_codec = match codec_id {
            CodecId::H264 => VideoCodec::H264,
            CodecId::HEVC => VideoCodec::Hevc,
            CodecId::VP9 => VideoCodec::Vp9,
            CodecId::AV1 => VideoCodec::Av1,
            _ => VideoCodec::Other,
        };

        let mut codec_context = codec::context::Context::from_parameters(codec_params)
            .context("Не удалось создать контекст декодера из параметров потока")?;

        // Многопоточное декодирование на случай программного фоллбэка (например, dav1d / hevc)
        unsafe {
            (*codec_context.as_mut_ptr()).thread_count = 0; // auto threads
        }

        // Попытка привязать аппаратный контекст VA-API / Vulkan / NVDEC / VDPAU
        let hw_device_ctx = Self::try_init_hw_device(&mut codec_context, hw_accel);

        let decoder = codec_context
            .decoder()
            .video()
            .context("Не удалось открыть видеодекодер libavcodec")?;

        info!(
            "Открыт нативный декодер ffmpeg-next: файл={:?}, кодек={:?} ({:?}), исходный размер={}x{}, целевой={}x{}@{}fps, hw_ctx_active={}",
            path,
            detected_codec,
            decoder.id(),
            decoder.width(),
            decoder.height(),
            width,
            height,
            target_fps,
            !hw_device_ctx.is_null()
        );

        Ok(Self {
            path,
            codec: detected_codec,
            hw_accel,
            scale_mode,
            target_fps: target_fps.max(1),
            width,
            height,
            paused: false,
            ictx,
            video_stream_index,
            decoder,
            hw_device_ctx,
            sws_scaler: None,
            last_src_format: None,
            decoded_frame: VideoFrame::empty(),
            sw_transfer_frame: VideoFrame::empty(),
            scaled_bgra_frame: VideoFrame::new(Pixel::BGRA, width, height),
            eof_flushed: false,
        })
    }

    /// Инициализирует `AVHWDeviceContext` (`VAAPI` / `VULKAN` / `CUDA` / `VDPAU`) и прикрепляет
    /// его к `AVCodecContext`. Если выбранный аппаратный бэкенд недоступен, возвращает `null_mut()`
    /// для бесшовного фоллбэка на CPU-декодер.
    fn try_init_hw_device(
        codec_ctx: &mut codec::context::Context,
        hw: HwAccelBackend,
    ) -> *mut AVBufferRef {
        if hw == HwAccelBackend::Software {
            return ptr::null_mut();
        }

        let candidates: &[AVHWDeviceType] = match hw {
            HwAccelBackend::Auto => &[
                AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
                AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
                AVHWDeviceType::AV_HWDEVICE_TYPE_VDPAU,
            ],
            HwAccelBackend::Vaapi => &[AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI],
            HwAccelBackend::Vulkan => &[AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN],
            HwAccelBackend::Nvdec => &[AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA],
            HwAccelBackend::Vdpau => &[AVHWDeviceType::AV_HWDEVICE_TYPE_VDPAU],
            HwAccelBackend::Software => &[],
        };

        let render_node = CString::new("/dev/dri/renderD128").unwrap();

        for &dev_type in candidates {
            let mut hw_ctx: *mut AVBufferRef = ptr::null_mut();
            let device_ptr = if dev_type == AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI
                && std::path::Path::new("/dev/dri/renderD128").exists()
            {
                render_node.as_ptr()
            } else {
                ptr::null()
            };

            let ret = unsafe {
                av_hwdevice_ctx_create(&mut hw_ctx, dev_type, device_ptr, ptr::null_mut(), 0)
            };

            if ret == 0 && !hw_ctx.is_null() {
                unsafe {
                    (*codec_ctx.as_mut_ptr()).hw_device_ctx = av_buffer_ref(hw_ctx);
                }
                info!("Активировано аппаратное ускорение FFmpeg: {:?}", dev_type);
                return hw_ctx;
            }
        }

        warn!(
            "Аппаратный декодер {:?} недоступен на данном GPU, используется многопоточный программный декодер libavcodec",
            hw
        );
        ptr::null_mut()
    }

    fn is_hw_pixel_format(pix_fmt: AVPixelFormat) -> bool {
        matches!(
            pix_fmt,
            AVPixelFormat::AV_PIX_FMT_VAAPI
                | AVPixelFormat::AV_PIX_FMT_VULKAN
                | AVPixelFormat::AV_PIX_FMT_CUDA
                | AVPixelFormat::AV_PIX_FMT_VDPAU
                | AVPixelFormat::AV_PIX_FMT_DRM_PRIME
        )
    }

    /// Обновляет целевое разрешение кадра при реконфигурации монитора в Niri.
    pub fn resize_output(&mut self, new_width: u32, new_height: u32) {
        if self.width == new_width && self.height == new_height {
            return;
        }
        self.width = new_width;
        self.height = new_height;
        self.scaled_bgra_frame = VideoFrame::new(Pixel::BGRA, new_width, new_height);
        self.sws_scaler = None;
    }

    /// Декодирует ровно один видеокадр и копирует его в `dest_bgra` (`XRGB8888` для `wl_shm`).
    pub fn decode_next_frame(&mut self, dest_bgra: &mut [u8]) -> Result<bool> {
        if self.paused {
            return Ok(false);
        }

        let expected_bytes = (self.width as usize) * (self.height as usize) * 4;
        if dest_bgra.len() < expected_bytes {
            anyhow::bail!("Целевой буфер wl_shm меньше размера кадра BGRA");
        }

        loop {
            // 1. Сначала проверяем, есть ли уже готовый кадр в очереди декодера
            if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                self.convert_decoded_frame_to_bgra(dest_bgra)?;
                return Ok(true);
            }

            // 2. Читаем следующий пакет из контейнера
            let mut packet_fed = false;
            for (stream, packet) in self.ictx.packets() {
                if stream.index() == self.video_stream_index {
                    self.decoder.send_packet(&packet)?;
                    packet_fed = true;
                    break;
                }
            }

            // 3. Если достигнут конец файла — сливаем хвост декодера и
            //    пересоздаём вход+декодер целиком. Практика показала, что
            //    `Input::seek(0, ..0)` в ffmpeg-next 9 после EOF возвращает
            //    Ok, но пайплайн декодера остаётся мёртвым — поэтому
            //    guarantee только один: свежий контейнер и свежий декодер.
            if !packet_fed {
                if !self.eof_flushed {
                    self.decoder.send_eof()?;
                    self.eof_flushed = true;
                    // Один кадр из drain-буфера декодера отдаем на этом же тике,
                    // чтобы не мигало на месте стыка лупа
                    if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                        self.convert_decoded_frame_to_bgra(dest_bgra)?;
                        return Ok(true);
                    }
                }
                // Вход исчерпан, декодер выдохся — перезапуск движка (луп видео)
                let old_paused = self.paused;
                *self = VideoWallpaperEngine::new(
                    self.path.clone(),
                    self.hw_accel,
                    self.scale_mode,
                    self.target_fps,
                    self.width,
                    self.height,
                )?;
                self.paused = old_paused;
                continue;
            }
        }
    }

    fn convert_decoded_frame_to_bgra(&mut self, dest_bgra: &mut [u8]) -> Result<()> {
        let raw_fmt = unsafe {
            std::mem::transmute::<i32, AVPixelFormat>((*self.decoded_frame.as_ptr()).format)
        };

        // Если кадр находится в видеопамяти VA-API / Vulkan / CUDA, переносим его в системную RAM (NV12/YUV420P)
        let src_frame: &VideoFrame = if Self::is_hw_pixel_format(raw_fmt) {
            let ret = unsafe {
                av_hwframe_transfer_data(
                    self.sw_transfer_frame.as_mut_ptr(),
                    self.decoded_frame.as_ptr(),
                    0,
                )
            };
            if ret < 0 {
                anyhow::bail!("Ошибка av_hwframe_transfer_data (код {})", ret);
            }
            &self.sw_transfer_frame
        } else {
            &self.decoded_frame
        };

        let src_fmt = src_frame.format();
        let src_w = src_frame.width();
        let src_h = src_frame.height();

        if self.sws_scaler.is_none() || self.last_src_format != Some((src_fmt, src_w, src_h)) {
            let flags = match self.scale_mode {
                ScaleMode::Fill | ScaleMode::Fit => SwsFlags::BILINEAR,
                ScaleMode::Stretch => SwsFlags::FAST_BILINEAR,
            };
            let scaler = SwsContext::get(
                src_fmt,
                src_w,
                src_h,
                Pixel::BGRA,
                self.width,
                self.height,
                flags,
            )
            .context("Не удалось инициализировать SwsContext для конвертации в BGRA")?;
            self.sws_scaler = Some(scaler);
            self.last_src_format = Some((src_fmt, src_w, src_h));
        }

        let scaler = self.sws_scaler.as_mut().unwrap();
        scaler.run(src_frame, &mut self.scaled_bgra_frame)?;

        // Копируем строки с учетом выравнивания linesize (stride) FFmpeg
        let src_stride = self.scaled_bgra_frame.stride(0);
        let dst_row_bytes = (self.width as usize) * 4;
        let data = self.scaled_bgra_frame.data(0);

        if src_stride == dst_row_bytes {
            let total = dst_row_bytes * (self.height as usize);
            dest_bgra[..total].copy_from_slice(&data[..total]);
        } else {
            for y in 0..(self.height as usize) {
                let s_off = y * src_stride;
                let d_off = y * dst_row_bytes;
                dest_bgra[d_off..d_off + dst_row_bytes]
                    .copy_from_slice(&data[s_off..s_off + dst_row_bytes]);
            }
        }

        Ok(())
    }

    pub fn set_paused(&mut self, pause: bool) {
        self.paused = pause;
    }
}

impl Drop for VideoWallpaperEngine {
    fn drop(&mut self) {
        if !self.hw_device_ctx.is_null() {
            unsafe {
                av_buffer_unref(&mut self.hw_device_ctx);
            }
        }
    }
}

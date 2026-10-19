//! Native hardware video decoder built on [`ffmpeg_next`] and [`ffmpeg_sys_next`].
//!
//! Decodes container streams (`.mp4`, `.mkv`, `.webm`) in-process without spawning external
//! `ffmpeg` CLI processes:
//! - Supports **H.264**, **HEVC (H.265)**, **VP9**, and **AV1**.
//! - Initializes hardware acceleration contexts (`VA-API`, `Vulkan`, `CUDA/NVDEC`, `VDPAU`) via
//!   `av_hwdevice_ctx_create` on the configured DRM render node, with automatic fallback to
//!   multithreaded software decoding (`libdav1d` / `libavcodec`).
//! - Seamlessly loops at EOF by seeking the demuxer to timestamp `0` and calling `decoder.flush()`
//!   (`avcodec_flush_buffers`), preserving the hardware decoder context without GPU reinitialization.
//! - Honors aspect-ratio scaling modes (`ScaleMode::Fill`, `ScaleMode::Fit`, `ScaleMode::Stretch`)
//!   when converting frames into the Wayland `XRGB8888` (`BGRA`) shared memory slot.

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
use tracing::{debug, info, warn};

/// Computed scaling dimensions for `ScaleMode::Fill`, `ScaleMode::Fit`, and `ScaleMode::Stretch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScaledGeometry {
    scaled_w: u32,
    scaled_h: u32,
    crop_x: usize,
    crop_y: usize,
    dst_x: usize,
    dst_y: usize,
    copy_w: usize,
    copy_h: usize,
}

impl ScaledGeometry {
    fn compute(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, mode: ScaleMode) -> Self {
        let sw = src_w.max(2) as f64;
        let sh = src_h.max(2) as f64;
        let dw = dst_w.max(2) as f64;
        let dh = dst_h.max(2) as f64;

        match mode {
            ScaleMode::Stretch => Self {
                scaled_w: dst_w.max(2),
                scaled_h: dst_h.max(2),
                crop_x: 0,
                crop_y: 0,
                dst_x: 0,
                dst_y: 0,
                copy_w: dst_w as usize,
                copy_h: dst_h as usize,
            },
            ScaleMode::Fit => {
                let scale = (dw / sw).min(dh / sh);
                let scaled_w = (((sw * scale).round() as u32) & !1).clamp(2, dst_w.max(2));
                let scaled_h = (((sh * scale).round() as u32) & !1).clamp(2, dst_h.max(2));
                let dst_x = ((dst_w.saturating_sub(scaled_w)) / 2) as usize;
                let dst_y = ((dst_h.saturating_sub(scaled_h)) / 2) as usize;
                Self {
                    scaled_w,
                    scaled_h,
                    crop_x: 0,
                    crop_y: 0,
                    dst_x,
                    dst_y,
                    copy_w: scaled_w.min(dst_w) as usize,
                    copy_h: scaled_h.min(dst_h) as usize,
                }
            }
            ScaleMode::Fill => {
                let scale = (dw / sw).max(dh / sh);
                let scaled_w = (((sw * scale).ceil() as u32) & !1).max(dst_w.max(2));
                let scaled_h = (((sh * scale).ceil() as u32) & !1).max(dst_h.max(2));
                let crop_x = ((scaled_w.saturating_sub(dst_w)) / 2) as usize;
                let crop_y = ((scaled_h.saturating_sub(dst_h)) / 2) as usize;
                Self {
                    scaled_w,
                    scaled_h,
                    crop_x,
                    crop_y,
                    dst_x: 0,
                    dst_y: 0,
                    copy_w: dst_w as usize,
                    copy_h: dst_h as usize,
                }
            }
        }
    }
}

pub struct VideoWallpaperEngine {
    pub path: PathBuf,
    pub codec: VideoCodec,
    pub hw_accel: HwAccelBackend,
    pub drm_render_node: String,
    pub scale_mode: ScaleMode,
    pub loop_playback: bool,
    pub target_fps: u32,
    pub width: u32,
    pub height: u32,
    pub paused: bool,
    ictx: format::context::Input,
    video_stream_index: usize,
    decoder: codec::decoder::Video,
    hw_device_ctx: *mut AVBufferRef,
    sws_scaler: Option<SwsContext>,
    last_scaler_key: Option<(Pixel, u32, u32, u32, u32)>,
    decoded_frame: VideoFrame,
    sw_transfer_frame: VideoFrame,
    scaled_bgra_frame: VideoFrame,
    eof_flushed: bool,
}

unsafe impl Send for VideoWallpaperEngine {}

impl VideoWallpaperEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: PathBuf,
        hw_accel: HwAccelBackend,
        scale_mode: ScaleMode,
        target_fps: u32,
        width: u32,
        height: u32,
        drm_render_node: String,
        loop_playback: bool,
    ) -> Result<Self> {
        ffmpeg_next::init().context("Failed to initialize FFmpeg libraries (ffmpeg-next)")?;

        let ictx = format::input(&path)
            .with_context(|| format!("Failed to open video container {:?}", path))?;

        let stream = ictx
            .streams()
            .best(MediaType::Video)
            .ok_or_else(|| anyhow::anyhow!("No video stream found in container {:?}", path))?;

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
            .context("Failed to create decoder context from stream parameters")?;

        // Enable multithreaded software decoding as a fallback (e.g. dav1d / ff_hevc).
        unsafe {
            (*codec_context.as_mut_ptr()).thread_count = 0;
        }

        let hw_device_ctx =
            Self::try_init_hw_device(&mut codec_context, hw_accel, &drm_render_node);

        let decoder = codec_context
            .decoder()
            .video()
            .context("Failed to open libavcodec video decoder")?;

        info!(
            "Opened native ffmpeg-next decoder: file={:?}, codec={:?} ({:?}), source={}x{}, target={}x{}@{}fps, scale={:?}, hw_ctx_active={}",
            path,
            detected_codec,
            decoder.id(),
            decoder.width(),
            decoder.height(),
            width,
            height,
            target_fps,
            scale_mode,
            !hw_device_ctx.is_null()
        );

        Ok(Self {
            path,
            codec: detected_codec,
            hw_accel,
            drm_render_node,
            scale_mode,
            loop_playback,
            target_fps: target_fps.max(1),
            width,
            height,
            paused: false,
            ictx,
            video_stream_index,
            decoder,
            hw_device_ctx,
            sws_scaler: None,
            last_scaler_key: None,
            decoded_frame: VideoFrame::empty(),
            sw_transfer_frame: VideoFrame::empty(),
            scaled_bgra_frame: VideoFrame::empty(),
            eof_flushed: false,
        })
    }

    /// Initializes an `AVHWDeviceContext` (`VAAPI` / `VULKAN` / `CUDA` / `VDPAU`) and attaches it to
    /// `AVCodecContext`. Returns `null_mut()` when unavailable so software decoding takes over.
    fn try_init_hw_device(
        codec_ctx: &mut codec::context::Context,
        hw: HwAccelBackend,
        drm_render_node: &str,
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

        let render_node_cstr = CString::new(drm_render_node)
            .unwrap_or_else(|_| CString::new("/dev/dri/renderD128").unwrap());
        let node_exists = std::path::Path::new(drm_render_node).exists();

        for &dev_type in candidates {
            let mut hw_ctx: *mut AVBufferRef = ptr::null_mut();
            let device_ptr = if dev_type == AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI && node_exists {
                render_node_cstr.as_ptr()
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
                info!("FFmpeg hardware acceleration enabled: {:?}", dev_type);
                return hw_ctx;
            }
        }

        warn!(
            "Hardware decoder {:?} unavailable on this GPU; falling back to multithreaded software decoder",
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

    /// Updates target dimensions after a Niri monitor resolution change.
    pub fn resize_output(&mut self, new_width: u32, new_height: u32) {
        if self.width == new_width && self.height == new_height {
            return;
        }
        self.width = new_width;
        self.height = new_height;
        self.sws_scaler = None;
        self.last_scaler_key = None;
    }

    /// Rewinds the video container and flushes the decoder buffers without destroying
    /// the `AVHWDeviceContext`. Falls back to reopening the container only if seeking fails.
    fn rewind_or_reopen(&mut self) -> Result<()> {
        if self.ictx.seek(0, ..0).is_ok() {
            self.decoder.flush();
            self.eof_flushed = false;
            debug!("Video stream rewound to start via seek(0) + decoder.flush()");
            return Ok(());
        }

        debug!("Container seek(0) unsupported; reopening container while preserving HW context");
        let old_paused = self.paused;
        *self = VideoWallpaperEngine::new(
            self.path.clone(),
            self.hw_accel,
            self.scale_mode,
            self.target_fps,
            self.width,
            self.height,
            self.drm_render_node.clone(),
            self.loop_playback,
        )?;
        self.paused = old_paused;
        Ok(())
    }

    /// Decodes one video frame and writes it into `dest_bgra` (`XRGB8888` layout for `wl_shm`).
    pub fn decode_next_frame(&mut self, dest_bgra: &mut [u8]) -> Result<bool> {
        if self.paused {
            return Ok(false);
        }

        let expected_bytes = (self.width as usize) * (self.height as usize) * 4;
        if dest_bgra.len() < expected_bytes {
            anyhow::bail!("Destination wl_shm buffer is smaller than the BGRA frame");
        }

        // Guard against infinite loops on empty/corrupt containers.
        for _attempt in 0..4 {
            if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                self.convert_decoded_frame_to_bgra(dest_bgra)?;
                return Ok(true);
            }

            let mut packet_fed = false;
            for (stream, packet) in self.ictx.packets() {
                if stream.index() == self.video_stream_index {
                    self.decoder.send_packet(&packet)?;
                    packet_fed = true;
                    break;
                }
            }

            if !packet_fed {
                if !self.eof_flushed {
                    let _ = self.decoder.send_eof();
                    self.eof_flushed = true;
                    if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                        self.convert_decoded_frame_to_bgra(dest_bgra)?;
                        return Ok(true);
                    }
                }

                if !self.loop_playback {
                    self.paused = true;
                    return Ok(false);
                }

                self.rewind_or_reopen()?;
            }
        }

        Ok(false)
    }

    fn convert_decoded_frame_to_bgra(&mut self, dest_bgra: &mut [u8]) -> Result<()> {
        let raw_fmt = unsafe {
            std::mem::transmute::<i32, AVPixelFormat>((*self.decoded_frame.as_ptr()).format)
        };

        // Transfer hardware surface (VA-API / Vulkan / CUDA) to system RAM if needed.
        let src_frame: &VideoFrame = if Self::is_hw_pixel_format(raw_fmt) {
            let ret = unsafe {
                av_hwframe_transfer_data(
                    self.sw_transfer_frame.as_mut_ptr(),
                    self.decoded_frame.as_ptr(),
                    0,
                )
            };
            if ret < 0 {
                anyhow::bail!("av_hwframe_transfer_data failed (code {})", ret);
            }
            &self.sw_transfer_frame
        } else {
            &self.decoded_frame
        };

        let src_fmt = src_frame.format();
        let src_w = src_frame.width();
        let src_h = src_frame.height();

        let geom = ScaledGeometry::compute(src_w, src_h, self.width, self.height, self.scale_mode);
        let scaler_key = (src_fmt, src_w, src_h, geom.scaled_w, geom.scaled_h);

        if self.sws_scaler.is_none() || self.last_scaler_key != Some(scaler_key) {
            let flags = match self.scale_mode {
                ScaleMode::Fill | ScaleMode::Fit => SwsFlags::BILINEAR,
                ScaleMode::Stretch => SwsFlags::FAST_BILINEAR,
            };
            let scaler = SwsContext::get(
                src_fmt,
                src_w,
                src_h,
                Pixel::BGRA,
                geom.scaled_w,
                geom.scaled_h,
                flags,
            )
            .context("Failed to initialize SwsContext for BGRA conversion")?;
            self.scaled_bgra_frame = VideoFrame::new(Pixel::BGRA, geom.scaled_w, geom.scaled_h);
            self.sws_scaler = Some(scaler);
            self.last_scaler_key = Some(scaler_key);
        }

        let scaler = self.sws_scaler.as_mut().unwrap();
        scaler.run(src_frame, &mut self.scaled_bgra_frame)?;

        let src_stride = self.scaled_bgra_frame.stride(0);
        let dst_row_bytes = (self.width as usize) * 4;
        let data = self.scaled_bgra_frame.data(0);

        // In Fit mode, clear letterbox/pillarbox regions to opaque black when dimensions differ.
        if self.scale_mode == ScaleMode::Fit
            && (geom.copy_w < self.width as usize || geom.copy_h < self.height as usize)
        {
            let total = dst_row_bytes * (self.height as usize);
            dest_bgra[..total].fill(0);
        }

        let copy_row_bytes = geom.copy_w * 4;
        let src_x_bytes = geom.crop_x * 4;
        let dst_x_bytes = geom.dst_x * 4;

        if geom.crop_x == 0
            && geom.crop_y == 0
            && geom.dst_x == 0
            && geom.dst_y == 0
            && src_stride == dst_row_bytes
            && copy_row_bytes == dst_row_bytes
            && geom.copy_h == self.height as usize
        {
            let total = dst_row_bytes * (self.height as usize);
            dest_bgra[..total].copy_from_slice(&data[..total]);
        } else {
            for y in 0..geom.copy_h {
                let s_off = (geom.crop_y + y) * src_stride + src_x_bytes;
                let d_off = (geom.dst_y + y) * dst_row_bytes + dst_x_bytes;
                if s_off + copy_row_bytes <= data.len()
                    && d_off + copy_row_bytes <= dest_bgra.len()
                {
                    dest_bgra[d_off..d_off + copy_row_bytes]
                        .copy_from_slice(&data[s_off..s_off + copy_row_bytes]);
                }
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

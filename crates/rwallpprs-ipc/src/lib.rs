//! Protocol used between:
//! 1. `rwallpprs` (CLI) <-> `rwallpprs-daemon` (control UNIX socket, u32 length prefix + bincode)
//! 2. `rwallpprs-daemon` <-> compiled Rust wallpaper binary (POSIX SHM + event pipe)

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Magic signature for wallpaper binaries embedded directly inside the CLI file.
pub const EMBEDDED_BIN_MAGIC: &[u8; 18] = b"RWALLPPRS_EMBED_V2";

/// Supported video codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoCodec {
    H264,
    Hevc,
    Vp9,
    Av1,
    Other,
}

/// Hardware video decoding backend for `ffmpeg-next` (`AVHWDeviceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum HwAccelBackend {
    #[default]
    Auto,
    Vaapi,
    Vulkan,
    Nvdec,
    Vdpau,
    Software,
}

/// Scale mode for images/video on the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ScaleMode {
    #[default]
    Fill,
    Fit,
    Stretch,
}

/// Run mode of a compiled Rust wallpaper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RustWallpaperMode {
    /// Continuous rendering at the configured FPS (animated procedural or WGPU shader wallpapers).
    #[default]
    Animated,
    /// Render exactly one frame at startup or on monitor resolution change (0% CPU/GPU while idle).
    Static,
    /// Render only when events arrive from the Niri compositor (strip scroll, workspace change, Overview).
    ReactiveNiri,
}

/// Power saving and auto-hide settings for Niri.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSavePolicy {
    pub enabled: bool,
    pub auto_hide_surface: bool,
    pub pause_on_fullscreen: bool,
    pub pause_on_maximized_column: bool,
    pub maximized_column_threshold: f32,
    pub wake_on_niri_overview: bool,
    pub battery_freeze_threshold: u8,
}

impl Default for PowerSavePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_hide_surface: true,
            pause_on_fullscreen: true,
            pause_on_maximized_column: true,
            maximized_column_threshold: 0.96,
            wake_on_niri_overview: true,
            battery_freeze_threshold: 25,
        }
    }
}

/// Control commands from the CLI to `rwallpprs-daemon`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    /// Set video wallpaper (decoded natively through `ffmpeg-next`).
    SetVideo {
        path: PathBuf,
        output: Option<String>,
        hw_accel: HwAccelBackend,
        fps_limit: u32,
        scale: ScaleMode,
    },
    /// Run an external compiled Rust wallpaper binary from disk.
    SetRustBin {
        bin_path: PathBuf,
        args: Vec<String>,
        output: Option<String>,
        mode: RustWallpaperMode,
        fps: u32,
    },
    /// Hand an ELF wallpaper binary embedded in the CLI straight into daemon memory (`memfd_create` on the daemon side).
    SetEmbeddedBin {
        name: String,
        elf_payload: Vec<u8>,
        args: Vec<String>,
        output: Option<String>,
        mode: RustWallpaperMode,
        fps: u32,
    },
    /// Update the auto-hide and power saving policy at runtime.
    UpdatePowerPolicy(PowerSavePolicy),
    /// Force a pause toggle.
    TogglePause,
    /// Request telemetry and status for every monitor.
    GetStatus,
    /// Shut the daemon down.
    Shutdown,
}

/// Daemon reply for the CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlResponse {
    Ok { message: String },
    Status(DaemonStatusReport),
    Error { reason: String },
}

/// Reliable framed message write over a socket (u32 LE length + bincode payload).
/// Carries both small commands and multi-megabyte ELF binaries in `SetEmbeddedBin`.
pub fn write_framed_msg<W: Write, T: Serialize>(mut writer: W, msg: &T) -> std::io::Result<()> {
    let bytes = bincode::serialize(msg)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let len = bytes.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

pub fn read_framed_msg<R: Read, T: for<'de> Deserialize<'de>>(mut reader: R) -> std::io::Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 256 * 1024 * 1024 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "IPC message size limit exceeded (256 MB)",
        ));
    }
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;
    bincode::deserialize(&payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatusReport {
    pub pid: u32,
    pub niri_socket_connected: bool,
    pub active_source: String,
    pub power_state: PowerRuntimeState,
    pub policy: PowerSavePolicy,
    pub outputs: Vec<OutputStatus>,
    pub niri_state: NiriViewportState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputStatus {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub refresh_mhz: u32,
    pub surface_visible: bool,
    pub current_fps: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerRuntimeState {
    ActiveRendering,
    FrozenStatic { reason: String },
    HiddenOccluded { reason: String },
    UserPaused,
}

/// Client state of the Niri compositor, synchronized through the official `niri-ipc` crate.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NiriViewportState {
    pub active_workspace_id: u64,
    pub active_workspace_idx: u32,
    pub overview_open: bool,
    pub has_fullscreen_window: bool,
    /// Fraction of screen width (0.0..=1.0) occupied by the active focused window on the current workspace.
    pub focused_window_width_ratio: f32,
    /// Fraction of screen width (0.0..=1.0) covered by windows on the active workspace
    /// (sum of Niri scrolling-strip column widths): a single fullscreen window, two or three
    /// windows that together fill the screen, or a stack of windows all approach 1.0.
    pub screen_coverage_ratio: f32,
    pub visible_windows_count: u32,
    /// Relative horizontal offset within the Niri tiling strip (column index + pixel shift).
    pub strip_scroll_offset: f64,
    pub cursor_uv: (f32, f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerEvent {
    RenderFrame {
        slot: u8,
        frame_index: u64,
        time_secs: f64,
        dt_secs: f32,
        width: u32,
        height: u32,
        niri: NiriViewportState,
    },
    Suspend,
    Resume,
    Terminate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerAck {
    FrameReady {
        slot: u8,
        frame_index: u64,
        damaged_rect: Option<(u32, u32, u32, u32)>,
    },
    FrameSkipped { frame_index: u64 },
}

/// Header at the start of the POSIX shared memory (`memfd`) buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmBufferHeader {
    pub magic: u32,
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub slot_size_bytes: u64,
}

impl ShmBufferHeader {
    pub const MAGIC: u32 = 0x5257_4C50; // "RWLP"
    pub const VERSION: u32 = 2;
    pub const HEADER_SIZE: usize = 64;

    pub fn new(width: u32, height: u32) -> Self {
        let stride = width * 4;
        let slot_size_bytes = (stride as u64) * (height as u64);
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            width,
            height,
            stride,
            slot_size_bytes,
        }
    }

    pub fn total_shm_size(&self) -> usize {
        Self::HEADER_SIZE + (self.slot_size_bytes as usize) * 2
    }

    pub fn slot_offset(&self, slot: u8) -> usize {
        Self::HEADER_SIZE + (slot as usize & 1) * (self.slot_size_bytes as usize)
    }
}

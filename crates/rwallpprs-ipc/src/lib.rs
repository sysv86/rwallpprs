//! Протокол взаимодействия между:
//! 1. `rwallpprs` (CLI) <-> `rwallpprs-daemon` (управляющий UNIX-сокет, префикс длины u32 + bincode/JSON)
//! 2. `rwallpprs-daemon` <-> Скомпилированный Rust-бинарник обоев (POSIX SHM + event pipe)

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Магическая подпись для бинарников обоев, вшитых прямо внутрь CLI-файла.
pub const EMBEDDED_BIN_MAGIC: &[u8; 18] = b"RWALLPPRS_EMBED_V2";

/// Поддерживаемые видеокодеки.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoCodec {
    H264,
    Hevc,
    Vp9,
    Av1,
    Other,
}

/// Аппаратный бэкенд декодирования видео для `ffmpeg-next` (`AVHWDeviceType`).
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

/// Режим масштабирования изображения/видео на мониторе.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ScaleMode {
    #[default]
    Fill,
    Fit,
    Stretch,
}

/// Режим работы скомпилированных Rust-обоев.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RustWallpaperMode {
    /// Постоянный рендер с заданным FPS (анимированные процедурные или WGPU-шейдерные обои).
    #[default]
    Animated,
    /// Рендер ровно 1 кадра при старте или смене разрешения монитора (0% CPU/GPU в простое).
    Static,
    /// Рендер только при приходе событий от композитора Niri (скролл ленты, смена воркспейса, Overview).
    ReactiveNiri,
}

/// Настройки энергосбережения и автоскрытия под Niri.
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

/// Управляющие команды от CLI к демону `rwallpprs-daemon`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    /// Установить видео-обои (декодируются нативно через `ffmpeg-next`).
    SetVideo {
        path: PathBuf,
        output: Option<String>,
        hw_accel: HwAccelBackend,
        fps_limit: u32,
        scale: ScaleMode,
    },
    /// Запустить внешний скомпилированный бинарник Rust-обоев с диска.
    SetRustBin {
        bin_path: PathBuf,
        args: Vec<String>,
        output: Option<String>,
        mode: RustWallpaperMode,
        fps: u32,
    },
    /// Передать вложенный в CLI ELF-бинарник обоев напрямую в память демона (`memfd_create` на стороне демона).
    SetEmbeddedBin {
        name: String,
        elf_payload: Vec<u8>,
        args: Vec<String>,
        output: Option<String>,
        mode: RustWallpaperMode,
        fps: u32,
    },
    /// Обновить политику автоскрытия и энергосбережения на лету.
    UpdatePowerPolicy(PowerSavePolicy),
    /// Принудительно переключить паузу.
    TogglePause,
    /// Запросить телеметрию и статус всех мониторов.
    GetStatus,
    /// Остановить демон.
    Shutdown,
}

/// Ответ демона для CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlResponse {
    Ok { message: String },
    Status(DaemonStatusReport),
    Error { reason: String },
}

/// Надёжная фреймированная отправка сообщения по сокету (u32 LE длина + bincode payload).
/// Позволяет передавать как легкие команды, так и многомегабайтные ELF-бинарники в `SetEmbeddedBin`.
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
            "Превышен лимит размера IPC-сообщения (256 МБ)",
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

/// Живое состояние композитора Niri, синхронизированное через официальный крейт `niri-ipc`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NiriViewportState {
    pub active_workspace_id: u64,
    pub active_workspace_idx: u32,
    pub overview_open: bool,
    pub has_fullscreen_window: bool,
    /// Доля ширины экрана (0.0..=1.0), занимаемая активным сфокусированным окном на текущем воркспейсе.
    pub focused_window_width_ratio: f32,
    /// Доля ширины экрана (0.0..=1.0), покрытая окнами активного воркспейса
    /// (сумма ширин колонок ленты Niri): 1 окно на весь экран, 2-3 окна, вместе
    /// закрывающие экран, или стопка окон дают значение около 1.0.
    pub screen_coverage_ratio: f32,
    pub visible_windows_count: u32,
    /// Относительное горизонтальное смещение в тайлинговой ленте Niri (индекс колонки + сдвиг пикселей).
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

/// Заголовок в начале POSIX Shared Memory (`memfd`) буфера.
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

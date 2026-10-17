use anyhow::Context;
use rwallpprs_ipc::{HwAccelBackend, PowerSavePolicy, RustWallpaperMode, ScaleMode};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DaemonConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub video: VideoConfig,
    #[serde(default)]
    pub rust_engine: RustEngineConfig,
    #[serde(default)]
    pub power_saving: PowerSavePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub layer: String,
    pub log_level: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            layer: "background".to_string(),
            log_level: "info".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    pub hw_accel: HwAccelBackend,
    pub drm_render_node: String,
    pub scale_mode: ScaleMode,
    pub fps_limit: u32,
    pub loop_playback: bool,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            hw_accel: HwAccelBackend::Auto,
            drm_render_node: "/dev/dri/renderD128".to_string(),
            scale_mode: ScaleMode::Fill,
            fps_limit: 60,
            loop_playback: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustEngineConfig {
    pub default_mode: RustWallpaperMode,
    pub target_fps: u32,
    pub double_buffering: bool,
    pub forward_niri_events: bool,
}

impl Default for RustEngineConfig {
    fn default() -> Self {
        Self {
            default_mode: RustWallpaperMode::Animated,
            target_fps: 60,
            double_buffering: true,
            forward_niri_events: true,
        }
    }
}

impl DaemonConfig {
    pub fn load_or_default(path: Option<&Path>) -> anyhow::Result<Self> {
        if let Some(p) = path {
            if p.exists() {
                let raw = std::fs::read_to_string(p)
                    .with_context(|| format!("Не удалось прочитать конфиг {:?}", p))?;
                let cfg: Self = toml::from_str(&raw)
                    .with_context(|| format!("Ошибка парсинга TOML в {:?}", p))?;
                return Ok(cfg);
            }
        }
        Ok(Self::default())
    }
}

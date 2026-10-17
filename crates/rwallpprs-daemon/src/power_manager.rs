//! Модуль энергосбережения и автоматического скрытия обоев.
//!
//! Опирается на:
//! 1. Реальную долю ширины активной колонки Niri (`focused_window_width_ratio >= policy.maximized_column_threshold`).
//! 2. Полноэкранный статус (`has_fullscreen_window`).
//! 3. Открытие режима Niri Overview (`overview_open`).
//! 4. Уровень заряда батареи из `/sys/class/power_supply/BAT*`.

use rwallpprs_ipc::{
    NiriViewportState, PowerRuntimeState, PowerSavePolicy, RustWallpaperMode,
};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

pub struct PowerManager {
    pub policy: PowerSavePolicy,
    pub user_paused: bool,
    last_battery_check: Instant,
    cached_on_battery: bool,
    cached_battery_percent: u8,
}

impl PowerManager {
    pub fn new(policy: PowerSavePolicy) -> Self {
        let mut mgr = Self {
            policy,
            user_paused: false,
            last_battery_check: Instant::now() - Duration::from_secs(60),
            cached_on_battery: false,
            cached_battery_percent: 100,
        };
        mgr.refresh_battery_sysfs();
        mgr
    }

    pub fn evaluate(
        &mut self,
        niri: &NiriViewportState,
        mode: RustWallpaperMode,
        first_frame_rendered: bool,
    ) -> PowerRuntimeState {
        if self.user_paused {
            return PowerRuntimeState::UserPaused;
        }

        if mode == RustWallpaperMode::Static && first_frame_rendered {
            return PowerRuntimeState::FrozenStatic {
                reason: "static_wallpaper_mode".into(),
            };
        }

        if !self.policy.enabled {
            return PowerRuntimeState::ActiveRendering;
        }

        if niri.overview_open && self.policy.wake_on_niri_overview {
            return PowerRuntimeState::ActiveRendering;
        }

        if self.policy.pause_on_fullscreen && niri.has_fullscreen_window {
            return PowerRuntimeState::HiddenOccluded {
                reason: "niri_fullscreen_window".into(),
            };
        }

        if self.policy.pause_on_maximized_column
            && !niri.overview_open
            && (niri.screen_coverage_ratio >= self.policy.maximized_column_threshold
                || niri.focused_window_width_ratio >= self.policy.maximized_column_threshold)
        {
            return PowerRuntimeState::HiddenOccluded {
                reason: "niri_screen_fully_covered".into(),
            };
        }

        if self.last_battery_check.elapsed() >= Duration::from_secs(15) {
            self.refresh_battery_sysfs();
        }
        if self.cached_on_battery
            && self.cached_battery_percent <= self.policy.battery_freeze_threshold
            && first_frame_rendered
        {
            return PowerRuntimeState::FrozenStatic {
                reason: "battery_below_threshold".into(),
            };
        }

        PowerRuntimeState::ActiveRendering
    }

    fn refresh_battery_sysfs(&mut self) {
        self.last_battery_check = Instant::now();
        let power_dir = Path::new("/sys/class/power_supply");
        if !power_dir.exists() {
            return;
        }

        if let Ok(entries) = fs::read_dir(power_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("BAT") {
                    let status_path = entry.path().join("status");
                    let cap_path = entry.path().join("capacity");

                    if let Ok(status) = fs::read_to_string(status_path) {
                        self.cached_on_battery = status.trim().eq_ignore_ascii_case("Discharging");
                    }
                    if let Ok(cap) = fs::read_to_string(cap_path) {
                        if let Ok(pct) = cap.trim().parse::<u8>() {
                            self.cached_battery_percent = pct;
                        }
                    }
                    break;
                }
            }
        }
    }
}

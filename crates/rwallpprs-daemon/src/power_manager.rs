//! Power saving and automatic wallpaper hiding.
//!
//! Decisions are based on:
//! 1. The real width fraction of the active Niri column (`focused_window_width_ratio >= policy.maximized_column_threshold`).
//! 2. Fullscreen state (`has_fullscreen_window`).
//! 3. Whether Niri Overview is open (`overview_open`).
//! 4. Battery charge level from `/sys/class/power_supply/BAT*`.
//!
//! [`power_state_hint`] turns a suppressed state into a one-line explanation for the
//! user, because "the wallpaper stopped moving" is otherwise indistinguishable from a
//! crash: the paused engine leaves its last frame on screen.

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

        if self.last_battery_check.elapsed() >= Duration::from_secs(self.policy.battery_poll_interval_sec.max(1) as u64) {
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

/// One-line explanation of a suppressed render state, for logs and CLI replies.
///
/// Returns `None` when the wallpaper is actually rendering.
pub fn power_state_hint(state: &PowerRuntimeState) -> Option<String> {
    let hint = match state {
        PowerRuntimeState::ActiveRendering => return None,
        PowerRuntimeState::UserPaused => {
            "paused by request; resume with `rwallpprs power --toggle-pause`".to_string()
        }
        PowerRuntimeState::FrozenStatic { reason }
        | PowerRuntimeState::HiddenOccluded { reason } => match reason.as_str() {
            "static_wallpaper_mode" => "static wallpaper mode renders a single frame; \
                 start it with `--mode animated` to keep it moving"
                .to_string(),
            "battery_below_threshold" => {
                "on battery below the configured limit; raise it with \
                 `rwallpprs power --battery-limit 0` (or disable power saving with \
                 `rwallpprs power --enabled off`)"
                    .to_string()
            }
            "niri_fullscreen_window" => {
                "a fullscreen window is open; disable with `rwallpprs power --pause-fullscreen off`"
                    .to_string()
            }
            "niri_screen_fully_covered" => {
                "the tiled windows cover the screen; disable with \
                 `rwallpprs power --pause-maximized off`"
                    .to_string()
            }
            other => format!("suppressed by the power saving policy ({other})"),
        },
    };
    Some(hint)
}

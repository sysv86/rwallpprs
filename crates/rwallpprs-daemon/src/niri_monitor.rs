//! Native integration with the **Niri** compositor through the official [`niri_ipc`] crate.
//!
//! - Connects via `niri_ipc::socket::Socket::connect()`.
//! - Subscribes to `niri_ipc::Request::EventStream`.
//! - Maintains an in-memory cache of workspaces (`HashMap<u64, Workspace>`) and windows (`HashMap<u64, Window>`).
//! - Computes monitor coverage (`focused_window_width_ratio`, `screen_coverage_ratio`) and strip
//!   scroll offset (`strip_scroll_offset`) against the real output dimensions reported by Wayland.

use niri_ipc::{socket::Socket, Event, Reply, Request, Window, Workspace};
use rwallpprs_ipc::NiriViewportState;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;
use tracing::{debug, info, warn};

pub struct NiriMonitor {
    state: Arc<Mutex<NiriViewportState>>,
    connected: Arc<AtomicBool>,
    output_width: Arc<AtomicU32>,
    output_height: Arc<AtomicU32>,
}

impl NiriMonitor {
    pub fn spawn(initial_width: u32, initial_height: u32) -> Self {
        let state = Arc::new(Mutex::new(NiriViewportState::default()));
        let connected = Arc::new(AtomicBool::new(false));
        let output_width = Arc::new(AtomicU32::new(initial_width.max(1)));
        let output_height = Arc::new(AtomicU32::new(initial_height.max(1)));

        let state_cl = Arc::clone(&state);
        let conn_cl = Arc::clone(&connected);
        let w_cl = Arc::clone(&output_width);
        let h_cl = Arc::clone(&output_height);

        thread::Builder::new()
            .name("niri-ipc-event-stream".into())
            .spawn(move || {
                Self::event_loop(state_cl, conn_cl, w_cl, h_cl);
            })
            .expect("Failed to start the niri-ipc thread");

        Self {
            state,
            connected,
            output_width,
            output_height,
        }
    }

    /// Updates the logical monitor dimensions after a Wayland `LayerSurfaceConfigure` event.
    pub fn set_output_dimensions(&self, width: u32, height: u32) {
        self.output_width.store(width.max(1), Ordering::Relaxed);
        self.output_height.store(height.max(1), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> NiriViewportState {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn event_loop(
        state: Arc<Mutex<NiriViewportState>>,
        connected: Arc<AtomicBool>,
        output_width: Arc<AtomicU32>,
        output_height: Arc<AtomicU32>,
    ) {
        let mut workspaces_cache: HashMap<u64, Workspace> = HashMap::new();
        let mut windows_cache: HashMap<u64, Window> = HashMap::new();

        loop {
            let mut socket = match Socket::connect() {
                Ok(s) => s,
                Err(e) => {
                    debug!("Waiting for Niri socket ($NIRI_SOCKET): {}", e);
                    connected.store(false, Ordering::Relaxed);
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }
            };

            match socket.send(Request::EventStream) {
                Ok(Reply::Ok(_)) => {
                    info!("Subscribed to niri_ipc::Request::EventStream");
                    connected.store(true, Ordering::Relaxed);
                }
                Ok(Reply::Err(err)) => {
                    warn!("Niri rejected the EventStream request: {}", err);
                    connected.store(false, Ordering::Relaxed);
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }
                Err(e) => {
                    warn!("Failed to send EventStream to the Niri socket: {}", e);
                    connected.store(false, Ordering::Relaxed);
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }
            }

            let mut read_event = socket.read_events();
            while let Ok(event) = read_event() {
                Self::process_event(
                    event,
                    &mut workspaces_cache,
                    &mut windows_cache,
                    &state,
                    output_width.load(Ordering::Relaxed),
                    output_height.load(Ordering::Relaxed),
                );
            }

            warn!("Niri event stream interrupted, reconnecting...");
            connected.store(false, Ordering::Relaxed);
            thread::sleep(Duration::from_secs(1));
        }
    }

    fn process_event(
        event: Event,
        workspaces: &mut HashMap<u64, Workspace>,
        windows: &mut HashMap<u64, Window>,
        state: &Arc<Mutex<NiriViewportState>>,
        out_w: u32,
        out_h: u32,
    ) {
        match event {
            Event::WorkspacesChanged {
                workspaces: new_list,
            } => {
                workspaces.clear();
                for ws in new_list {
                    workspaces.insert(ws.id, ws);
                }
            }
            Event::WorkspaceActivated { id, focused } => {
                for ws in workspaces.values_mut() {
                    if ws.id == id {
                        ws.is_active = true;
                        ws.is_focused = focused;
                    } else if focused {
                        ws.is_focused = false;
                    }
                }
            }
            Event::WindowsChanged { windows: new_wins } => {
                windows.clear();
                for w in new_wins {
                    windows.insert(w.id, w);
                }
            }
            Event::WindowOpenedOrChanged { window } => {
                if window.is_focused {
                    for w in windows.values_mut() {
                        w.is_focused = false;
                    }
                }
                windows.insert(window.id, window);
            }
            Event::WindowClosed { id } => {
                windows.remove(&id);
            }
            Event::WindowFocusChanged { id } => {
                for w in windows.values_mut() {
                    w.is_focused = Some(w.id) == id;
                }
            }
            Event::WindowLayoutsChanged { changes } => {
                for (id, layout) in changes {
                    if let Some(w) = windows.get_mut(&id) {
                        w.layout = layout;
                    }
                }
            }
            Event::OverviewOpenedOrClosed { is_open } => {
                let mut guard = state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.overview_open = is_open;
                return;
            }
            _ => {}
        }

        let mut guard = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(active_ws) = workspaces
            .values()
            .find(|ws| ws.is_focused)
            .or_else(|| workspaces.values().find(|ws| ws.is_active))
        {
            guard.active_workspace_id = active_ws.id;
            guard.active_workspace_idx = active_ws.idx as u32;
        }

        let active_ws_id = guard.active_workspace_id;
        let mut count_on_ws = 0u32;
        let mut has_fullscreen = false;
        let mut focused_width_ratio = 0.0f32;
        let mut focused_col_idx: Option<usize> = None;

        let mut column_widths: HashMap<usize, f32> = HashMap::new();
        let mut no_layout_width = 0.0f32;

        for win in windows.values() {
            if win.workspace_id == Some(active_ws_id) || active_ws_id == 0 {
                count_on_ws += 1;
                let tile_w = win.layout.tile_size.0 as f32;
                let tile_h = win.layout.tile_size.1 as f32;

                if tile_w >= (out_w as f32 * 0.995) && tile_h >= (out_h as f32 * 0.995) {
                    has_fullscreen = true;
                }

                if win.is_floating {
                    no_layout_width = no_layout_width.max(tile_w.min(out_w as f32));
                } else if let Some((col_idx, _row_idx)) = win.layout.pos_in_scrolling_layout {
                    let entry = column_widths.entry(col_idx).or_insert(0.0);
                    *entry = entry.max(tile_w);
                } else {
                    no_layout_width = no_layout_width.max(tile_w.min(out_w as f32));
                }

                if win.is_focused {
                    focused_width_ratio = (tile_w / (out_w.max(1) as f32)).clamp(0.0, 1.0);
                    if let Some((col_idx, _row_idx)) = win.layout.pos_in_scrolling_layout {
                        focused_col_idx = Some(col_idx);
                    }
                }
            }
        }

        if let Some(target_col) = focused_col_idx {
            // Compute strip offset from accumulated column widths to the left of the focused column.
            let preceding_px: f64 = column_widths
                .iter()
                .filter(|(&col, _)| col < target_col)
                .map(|(_, &w)| (w as f64).max(320.0) * 0.35)
                .sum();
            guard.strip_scroll_offset = if preceding_px > 0.0 {
                preceding_px
            } else {
                (target_col as f64) * 320.0
            };
        }

        let cols_total: f32 = column_widths.values().copied().sum::<f32>() + no_layout_width;
        let screen_coverage_ratio = (cols_total / (out_w.max(1) as f32)).clamp(0.0, 1.0);

        guard.visible_windows_count = count_on_ws;
        guard.has_fullscreen_window = has_fullscreen;
        guard.focused_window_width_ratio = focused_width_ratio;
        guard.screen_coverage_ratio = screen_coverage_ratio;
    }
}

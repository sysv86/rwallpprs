//! Нативная интеграция с композитором **Niri** через официальный крейт [`niri_ipc`].
//!
//! Никакого ручного склеивания JSON-строк и никаких захардкоженных разрешений `1900x1060`:
//! - Подключается через `niri_ipc::socket::Socket::connect()`.
//! - Отправляет типизированный `niri_ipc::Request::EventStream`.
//! - Поддерживает полный кеш состояния воркспейсов (`HashMap<u64, Workspace>`) и окон (`HashMap<u64, Window>`).
//! - Рассчитывает долю перекрытия монитора (`focused_window_width_ratio`) относительно реальной ширины
//!   монитора, полученной от Wayland `zwlr_layer_surface_v1::configure`.

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
            .expect("Не удалось запустить поток niri-ipc");

        Self {
            state,
            connected,
            output_width,
            output_height,
        }
    }

    /// Обновляет реальные логические габариты монитора при событии `LayerSurfaceConfigure`.
    pub fn set_output_dimensions(&self, width: u32, height: u32) {
        self.output_width.store(width.max(1), Ordering::Relaxed);
        self.output_height.store(height.max(1), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> NiriViewportState {
        self.state.lock().unwrap().clone()
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
                    debug!("Ожидание сокета Niri ($NIRI_SOCKET): {}", e);
                    connected.store(false, Ordering::Relaxed);
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }
            };

            match socket.send(Request::EventStream) {
                Ok(Reply::Ok(_)) => {
                    info!("Успешная подписка на niri_ipc::Request::EventStream");
                    connected.store(true, Ordering::Relaxed);
                }
                Ok(Reply::Err(err)) => {
                    warn!("Niri отклонил запрос EventStream: {}", err);
                    connected.store(false, Ordering::Relaxed);
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }
                Err(e) => {
                    warn!("Ошибка отправки EventStream в сокет Niri: {}", e);
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

            warn!("Поток событий Niri прерван, переподключение...");
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
                let mut guard = state.lock().unwrap();
                guard.overview_open = is_open;
                return;
            }
            _ => {}
        }

        // Пересчитываем агрегированное состояние вьюпорта Niri
        let mut guard = state.lock().unwrap();

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

        // Покрытие экрана считаем по колонкам ленты Niri: ширина колонки —
        // это максимальная ширина окна в ней (стопки окон делят одну колонку).
        // Сумма ширин колонок, отнесённая к ширине монитора, корректно описывает
        // и одно окно на весь экран, и 2-3 окна, вместе занимающие весь экран.
        let mut column_widths: HashMap<usize, f32> = HashMap::new();
        let mut no_layout_width = 0.0f32;

        for win in windows.values() {
            if win.workspace_id == Some(active_ws_id) || active_ws_id == 0 {
                count_on_ws += 1;
                let tile_w = win.layout.tile_size.0 as f32;
                let tile_h = win.layout.tile_size.1 as f32;

                // Проверяем, занимает ли окно 100% физического/логического экрана (Fullscreen в Niri)
                if tile_w >= (out_w as f32 * 0.995) && tile_h >= (out_h as f32 * 0.995) {
                    has_fullscreen = true;
                }

                // Плавающие окна не тайлятся в колонки — учитываем их отдельно,
                // чтобы они не «склеивались» в одну колонку с тайлом по row-индексу.
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
                        guard.strip_scroll_offset = (col_idx as f64) * 320.0;
                    }
                }
            }
        }

        // Колонки ленты идут вплотную: если сумма их ширин покрывает монитор,
        // обои не видны ни при каком горизонтальном скролле.
        let cols_total: f32 = column_widths.values().copied().sum::<f32>() + no_layout_width;
        let screen_coverage_ratio = (cols_total / (out_w.max(1) as f32)).clamp(0.0, 1.0);

        guard.visible_windows_count = count_on_ws;
        guard.has_fullscreen_window = has_fullscreen;
        guard.focused_window_width_ratio = focused_width_ratio;
        guard.screen_coverage_ratio = screen_coverage_ratio;
    }
}

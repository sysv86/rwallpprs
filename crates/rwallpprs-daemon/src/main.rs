mod config;
mod niri_monitor;
mod power_manager;
mod rust_runner;
mod video_engine;
mod wayland_layer;

use clap::Parser;
use config::DaemonConfig;
use niri_monitor::NiriMonitor;
use power_manager::PowerManager;
use rust_runner::RustWallpaperRunner;
use rwallpprs_ipc::{
    read_framed_msg, write_framed_msg, ControlCommand, ControlResponse, DaemonStatusReport,
    OutputStatus, PowerRuntimeState, RustWallpaperMode,
};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};
use video_engine::VideoWallpaperEngine;
use wayland_layer::WaylandLayerContext;

#[derive(Parser, Debug)]
#[command(name = "rwallpprs-daemon", about = "RWallpprs v0.2.0-beta.1 Wayland Daemon for Niri")]
struct Args {
    #[arg(short, long)]
    config: Option<PathBuf>,
}

enum ActiveEngine {
    None,
    Video(VideoWallpaperEngine),
    RustBin(RustWallpaperRunner),
}

pub fn control_socket_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(runtime_dir).join("rwallpprs.sock")
}

pub fn run_daemon(config_path: Option<PathBuf>) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = DaemonConfig::load_or_default(config_path.as_deref())?;
    info!("Запуск демона RWallpprs v0.2.0-beta.1 под композитор Niri...");

    let sock_path = control_socket_path();
    if sock_path.exists() {
        let _ = std::fs::remove_file(&sock_path);
    }
    let listener = UnixListener::bind(&sock_path)?;
    listener.set_nonblocking(true)?;
    info!("Управляющий сокет открыт: {:?}", sock_path);

    let mut wayland_session = match WaylandLayerContext::connect_and_init() {
        Ok((conn, queue, ctx)) => {
            info!(
                "Wayland wlr-layer-shell инициализирован ({}x{})",
                ctx.width, ctx.height
            );
            Some((conn, queue, ctx))
        }
        Err(e) => {
            warn!(
                "WAYLAND_DISPLAY недоступен ({}). Работа в режиме ожидания.",
                e
            );
            None
        }
    };

    let (init_w, init_h) = wayland_session
        .as_ref()
        .map(|(_, _, c)| (c.width, c.height))
        .unwrap_or((1920, 1080));

    let niri_monitor = NiriMonitor::spawn(init_w, init_h);
    let mut power_mgr = PowerManager::new(cfg.power_saving.clone());
    let mut active_engine = ActiveEngine::None;
    let mut first_frame_rendered = false;
    let mut frame_scratch: Vec<u8> = Vec::new();
    let mut running = true;

    while running {
        let frame_start = Instant::now();

        // 1. Прием команд от CLI (включая потоковую загрузку вложенного ELF-бинарника)
        while let Ok((stream, _)) = listener.accept() {
            let _ = stream.set_nonblocking(false);
            if let Err(e) = handle_cli_client(
                stream,
                &mut active_engine,
                &mut power_mgr,
                &niri_monitor,
                &wayland_session,
                &mut first_frame_rendered,
                &mut running,
            ) {
                error!("Ошибка обработки команды CLI: {}", e);
            }
        }

        // 2. Диспетчеризация событий Wayland и отслеживание смены разрешения монитора
        if let Some((ref conn, ref mut queue, ref mut wl_ctx)) = wayland_session {
            // roundtrip читает входящие ответы композитора (включая wl_buffer.release),
            // иначе SlotPool не освобождает слоты и память утекает OOM-сессию
            let _ = conn.flush();
            let _ = queue.roundtrip(wl_ctx);
            if wl_ctx.resolution_changed {
                wl_ctx.resolution_changed = false;
                niri_monitor.set_output_dimensions(wl_ctx.width, wl_ctx.height);
                if let ActiveEngine::Video(ref mut video) = active_engine {
                    video.resize_output(wl_ctx.width, wl_ctx.height);
                }
                first_frame_rendered = false;
            }
        }

        // 3. Оценка состояния Niri и политики энергосбережения
        let niri_state = niri_monitor.snapshot();
        let current_mode = match &active_engine {
            ActiveEngine::RustBin(runner) => runner.mode,
            ActiveEngine::Video(_) => RustWallpaperMode::Animated,
            ActiveEngine::None => RustWallpaperMode::Static,
        };

        let power_state = power_mgr.evaluate(&niri_state, current_mode, first_frame_rendered);

        match power_state {
            PowerRuntimeState::HiddenOccluded { .. } => {
                match &mut active_engine {
                    ActiveEngine::Video(v) => v.set_paused(true),
                    ActiveEngine::RustBin(r) => r.set_suspended(true),
                    ActiveEngine::None => {}
                }
                if power_mgr.policy.auto_hide_surface {
                    if let Some((_, _, ref mut wl_ctx)) = wayland_session {
                        wl_ctx.hide_surface_for_powersave();
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            PowerRuntimeState::FrozenStatic { .. } | PowerRuntimeState::UserPaused => {
                match &mut active_engine {
                    ActiveEngine::Video(v) => v.set_paused(true),
                    ActiveEngine::RustBin(r) => r.set_suspended(true),
                    ActiveEngine::None => {}
                }
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            PowerRuntimeState::ActiveRendering => {
                match &mut active_engine {
                    ActiveEngine::Video(v) => v.set_paused(false),
                    ActiveEngine::RustBin(r) => r.set_suspended(false),
                    ActiveEngine::None => {}
                }
            }
        }

        // 4. Декодирование видеокадра через ffmpeg-next или запрос кадра у Rust-бинарника
        let target_fps = match &mut active_engine {
            ActiveEngine::None => 15,
            ActiveEngine::Video(video) => {
                let needed = (video.width as usize) * (video.height as usize) * 4;
                if frame_scratch.len() != needed {
                    frame_scratch.resize(needed, 0);
                }
                match video.decode_next_frame(&mut frame_scratch) {
                    Ok(true) => {
                        if let Some((_, _, ref mut wl_ctx)) = wayland_session {
                            if let Err(e) = wl_ctx.present_frame(&frame_scratch) {
                                error!("Ошибка вывода видеокадра на Wayland: {}", e);
                            }
                        }
                        first_frame_rendered = true;
                    }
                    Ok(false) => {}
                    Err(e) => error!("Ошибка декодирования кадра ffmpeg-next: {}", e),
                }
                video.target_fps
            }
            ActiveEngine::RustBin(runner) => {
                let fps = runner.target_fps;
                match runner.request_frame(niri_state) {
                    Ok(Some(frame_bytes)) => {
                        if let Some((_, _, ref mut wl_ctx)) = wayland_session {
                            if let Err(e) = wl_ctx.present_frame(frame_bytes) {
                                error!("Ошибка вывода кадра Rust-обоев на Wayland: {}", e);
                            }
                        }
                        first_frame_rendered = true;
                    }
                    Ok(None) => {}
                    Err(e) => error!("Ошибка получения кадра от Rust-обоев: {}", e),
                }
                fps
            }
        };

        let frame_budget = Duration::from_secs_f64(1.0 / (target_fps.max(1) as f64));
        let elapsed = frame_start.elapsed();
        if elapsed < frame_budget {
            std::thread::sleep(frame_budget - elapsed);
        }
    }

    let _ = std::fs::remove_file(&sock_path);
    info!("Демон RWallpprs штатно завершил работу.");
    Ok(())
}

fn handle_cli_client(
    mut stream: UnixStream,
    active_engine: &mut ActiveEngine,
    power_mgr: &mut PowerManager,
    niri_monitor: &NiriMonitor,
    wayland_session: &Option<(
        wayland_client::Connection,
        wayland_client::EventQueue<WaylandLayerContext>,
        WaylandLayerContext,
    )>,
    first_frame_rendered: &mut bool,
    running: &mut bool,
) -> anyhow::Result<()> {
    let (width, height, surface_visible) = wayland_session
        .as_ref()
        .map(|(_, _, ctx)| (ctx.width, ctx.height, !ctx.surface_hidden))
        .unwrap_or((1920, 1080, false));

    let cmd: ControlCommand = read_framed_msg(&mut stream)?;
    let resp = match cmd {
        ControlCommand::SetVideo {
            path,
            hw_accel,
            fps_limit,
            scale,
            ..
        } => match VideoWallpaperEngine::new(path.clone(), hw_accel, scale, fps_limit, width, height)
        {
            Ok(engine) => {
                let codec = engine.codec;
                *active_engine = ActiveEngine::Video(engine);
                *first_frame_rendered = false;
                ControlResponse::Ok {
                    message: format!(
                        "Запущен нативный декодер ffmpeg-next для {:?} (кодек {:?}, hw={:?}, {}x{}@{}fps)",
                        path, codec, hw_accel, width, height, fps_limit
                    ),
                }
            }
            Err(e) => ControlResponse::Error {
                reason: format!("{:#}", e),
            },
        },
        ControlCommand::SetRustBin {
            bin_path,
            args,
            mode,
            fps,
            ..
        } => match RustWallpaperRunner::spawn_from_path(
            bin_path.clone(),
            args,
            mode,
            fps,
            width,
            height,
        ) {
            Ok(runner) => {
                let pid = runner.pid();
                *active_engine = ActiveEngine::RustBin(runner);
                *first_frame_rendered = false;
                ControlResponse::Ok {
                    message: format!(
                        "Запущены нативные Rust-обои {:?} (PID={}, режим={:?}, {}x{}@{}fps)",
                        bin_path, pid, mode, width, height, fps
                    ),
                }
            }
            Err(e) => ControlResponse::Error {
                reason: format!("{:#}", e),
            },
        },
        ControlCommand::SetEmbeddedBin {
            name,
            elf_payload,
            args,
            mode,
            fps,
            ..
        } => match RustWallpaperRunner::spawn_from_embedded_elf(
            name.clone(),
            &elf_payload,
            args,
            mode,
            fps,
            width,
            height,
        ) {
            Ok(runner) => {
                let pid = runner.pid();
                *active_engine = ActiveEngine::RustBin(runner);
                *first_frame_rendered = false;
                ControlResponse::Ok {
                    message: format!(
                        "Вложенный бинарник '{}' ({} КБ) загружен в персистентный memfd демона и запущен (PID={}, режим={:?})",
                        name,
                        elf_payload.len() / 1024,
                        pid,
                        mode
                    ),
                }
            }
            Err(e) => ControlResponse::Error {
                reason: format!("{:#}", e),
            },
        },
        ControlCommand::UpdatePowerPolicy(new_policy) => {
            power_mgr.policy = new_policy;
            ControlResponse::Ok {
                message: "Политика энергосбережения и автоскрытия Niri обновлена".into(),
            }
        }
        ControlCommand::TogglePause => {
            power_mgr.user_paused = !power_mgr.user_paused;
            ControlResponse::Ok {
                message: format!("Пауза переключена: user_paused={}", power_mgr.user_paused),
            }
        }
        ControlCommand::GetStatus => {
            let niri_state = niri_monitor.snapshot();
            let (source_desc, current_fps, mode) = match active_engine {
                ActiveEngine::None => ("none".to_string(), 0.0, RustWallpaperMode::Static),
                ActiveEngine::Video(v) => (
                    format!("ffmpeg-next:{:?} ({:?}, {:?})", v.path, v.codec, v.hw_accel),
                    v.target_fps as f32,
                    RustWallpaperMode::Animated,
                ),
                ActiveEngine::RustBin(r) => (
                    format!("rust-bin:{} (PID={}, {:?})", r.display_name, r.pid(), r.mode),
                    r.target_fps as f32,
                    r.mode,
                ),
            };
            let power_state = power_mgr.evaluate(&niri_state, mode, *first_frame_rendered);
            ControlResponse::Status(DaemonStatusReport {
                pid: std::process::id(),
                niri_socket_connected: niri_monitor.is_connected(),
                active_source: source_desc,
                power_state,
                policy: power_mgr.policy.clone(),
                outputs: vec![OutputStatus {
                    name: "Niri-Primary-Output".into(),
                    width,
                    height,
                    refresh_mhz: 60_000,
                    surface_visible,
                    current_fps,
                }],
                niri_state,
            })
        }
        ControlCommand::Shutdown => {
            *running = false;
            ControlResponse::Ok {
                message: "Демон RWallpprs останавливается".into(),
            }
        }
    };

    write_framed_msg(&mut stream, &resp)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_daemon(args.config)
}

mod config;
mod niri_monitor;
mod power_manager;
mod rust_runner;
mod video_engine;
mod wayland_layer;

use anyhow::Context as _;
use clap::Parser;
use config::DaemonConfig;
use niri_monitor::NiriMonitor;
use power_manager::{power_state_hint, PowerManager};
use rust_runner::RustWallpaperRunner;
use rustix::{
    fd::{AsRawFd, OwnedFd},
    fs::{ftruncate, memfd_create, MemfdFlags},
};
use rwallpprs_ipc::{
    read_framed_msg, write_framed_msg, ControlCommand, ControlResponse, DaemonStatusReport,
    OutputStatus, PowerRuntimeState, RustWallpaperMode, ShmBufferHeader,
};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use video_engine::VideoWallpaperEngine;
use wayland_layer::WaylandLayerContext;

#[derive(Parser, Debug)]
#[command(
    name = "rwallpprs-daemon",
    about = "RWallpprs v0.2.0-beta.2 Wayland daemon for Niri"
)]
struct Args {
    #[arg(short, long)]
    config: Option<PathBuf>,
}

enum ActiveEngine {
    None,
    Video(VideoWallpaperEngine),
    RustBin(RustWallpaperRunner),
}

/// Daemon memfd for zero-copy direct presentation of the video engine.
struct VideoDirectRing {
    shm_fd: OwnedFd,
    width: u32,
    height: u32,
}

impl VideoDirectRing {
    fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        let shm_fd = memfd_create("rwallpprs-video-direct", MemfdFlags::CLOEXEC)
            .context("rustix::fs::memfd_create failed for the video SHM ring")?;
        ftruncate(
            &shm_fd,
            ShmBufferHeader::new(width, height).total_shm_size() as u64,
        )
        .context("rustix::fs::ftruncate failed for the video SHM ring")?;
        Ok(Self {
            shm_fd,
            width,
            height,
        })
    }

    fn fits(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }
}

pub fn control_socket_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(runtime_dir).join("rwallpprs.sock")
}

/// Minimum number of main-loop ticks between retries of the video direct presenter.
const VIDEO_DIRECT_RETRY_TICKS: u32 = 30;

fn resize_active_engine(active_engine: &mut ActiveEngine, width: u32, height: u32) {
    match active_engine {
        ActiveEngine::Video(ref mut video) => {
            video.resize_output(width, height);
        }
        ActiveEngine::RustBin(ref mut runner) => {
            if let Err(e) = runner.resize_output(width, height) {
                error!("Failed to resize Rust wallpaper runner to {}x{}: {}", width, height, e);
            }
        }
        ActiveEngine::None => {}
    }
}

pub fn run_daemon(config_path: Option<PathBuf>) -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    std::panic::set_hook(Box::new(|info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        error!("PANIC in rwallpprs-daemon: {info}\n{backtrace}");
    }));

    let cfg = DaemonConfig::load_or_default(config_path.as_deref())?;
    info!("Starting RWallpprs v0.2.0-beta.2 daemon under the Niri compositor...");

    let sock_path = control_socket_path();
    if sock_path.exists() {
        let _ = std::fs::remove_file(&sock_path);
    }
    let listener = UnixListener::bind(&sock_path)?;
    listener.set_nonblocking(true)?;
    info!("Control socket open: {:?}", sock_path);

    let mut wayland_session = match WaylandLayerContext::connect_and_init() {
        Ok((conn, queue, ctx)) => {
            info!(
                "Wayland wlr-layer-shell initialized ({}x{})",
                ctx.width, ctx.height
            );
            Some((conn, queue, ctx))
        }
        Err(e) => {
            warn!(
                "WAYLAND_DISPLAY unavailable ({}). Running in standby mode.",
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
    let mut video_direct_ring: Option<VideoDirectRing> = None;
    let mut video_direct_backoff: u32 = 0;
    let mut video_slot: u8 = 0;
    let mut running = true;
    let mut last_power_reason: Option<String> = None;

    while running {
        let frame_start = Instant::now();

        // 1. Accept commands from the CLI
        while let Ok((stream, _)) = listener.accept() {
            let _ = stream.set_nonblocking(false);
            if let Err(e) = handle_cli_client(
                stream,
                &cfg,
                &mut active_engine,
                &mut power_mgr,
                &niri_monitor,
                &wayland_session,
                &mut first_frame_rendered,
                &mut running,
            ) {
                error!("Failed to handle a CLI command: {}", e);
            }
        }

        // 2. Dispatch Wayland events and detect monitor resolution changes
        let mut wl_dead = false;
        if let Some((ref conn, ref mut queue, ref mut wl_ctx)) = wayland_session {
            let _ = conn.flush();
            if let Err(e) = queue.roundtrip(wl_ctx) {
                error!("Wayland connection lost ({}), reconnect scheduled", e);
                wl_dead = true;
            } else if wl_ctx.resolution_changed {
                wl_ctx.resolution_changed = false;
                niri_monitor.set_output_dimensions(wl_ctx.width, wl_ctx.height);
                resize_active_engine(&mut active_engine, wl_ctx.width, wl_ctx.height);
                first_frame_rendered = false;
            }
        }
        if wl_dead {
            wayland_session = None;
        }
        if wayland_session.is_none() {
            match WaylandLayerContext::connect_and_init() {
                Ok((conn, queue, ctx)) => {
                    info!(
                        "Wayland reconnected, layer recreated ({}x{})",
                        ctx.width, ctx.height
                    );
                    niri_monitor.set_output_dimensions(ctx.width, ctx.height);
                    resize_active_engine(&mut active_engine, ctx.width, ctx.height);
                    first_frame_rendered = false;
                    wayland_session = Some((conn, queue, ctx));
                }
                Err(e) => {
                    debug!("Waiting for a Wayland connection: {}", e);
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
            }
        }

        // 3. Evaluate Niri state and the power saving policy
        let niri_state = niri_monitor.snapshot();
        let current_mode = match &active_engine {
            ActiveEngine::RustBin(runner) => runner.mode,
            ActiveEngine::Video(_) => RustWallpaperMode::Animated,
            ActiveEngine::None => RustWallpaperMode::Static,
        };

        let power_state = power_mgr.evaluate(&niri_state, current_mode, first_frame_rendered);
        log_power_transition(&mut last_power_reason, &power_state);

        match power_state {
            PowerRuntimeState::HiddenOccluded { .. } => {
                match &mut active_engine {
                    ActiveEngine::Video(v) => v.set_paused(true),
                    ActiveEngine::RustBin(r) => r.set_suspended(true),
                    ActiveEngine::None => {}
                }
                if power_mgr.policy.auto_hide_surface {
                    if let Some((_, _, ref mut wl_ctx)) = wayland_session {
                        wl_ctx.mark_hidden_for_powersave();
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

        // 4. Decode a video frame through ffmpeg-next or request a frame from the Rust binary
        let target_fps = match &mut active_engine {
            ActiveEngine::None => 15,
            ActiveEngine::Video(video) => {
                let mut direct_path_used = false;

                if let Some((_, ref queue, ref mut wl_ctx)) = wayland_session {
                    let ring_src = video_direct_ring
                        .as_ref()
                        .map(|ring| (ring.shm_fd.as_raw_fd(), ring.width, ring.height));
                    wl_ctx.reset_direct_if_stale(ring_src);

                    if video_direct_backoff > 0 {
                        video_direct_backoff -= 1;
                    } else if wl_ctx.direct.is_none()
                        && video.width == wl_ctx.width
                        && video.height == wl_ctx.height
                    {
                        let ring_ok = video_direct_ring
                            .as_ref()
                            .map(|ring| ring.fits(video.width, video.height))
                            .unwrap_or(false);
                        if !ring_ok {
                            video_direct_ring = None;
                            match VideoDirectRing::new(video.width, video.height) {
                                Ok(ring) => video_direct_ring = Some(ring),
                                Err(e) => {
                                    video_direct_backoff = VIDEO_DIRECT_RETRY_TICKS;
                                    warn!(
                                        "Failed to create daemon video SHM memfd: {}. Retrying in {} ticks; fallback: copy through present_frame",
                                        e, VIDEO_DIRECT_RETRY_TICKS
                                    );
                                }
                            }
                        }
                        if video_direct_backoff == 0 {
                            if let Some(ring) = video_direct_ring.as_ref() {
                                let qh = queue.handle();
                                match wl_ctx.init_direct_shm(
                                    &qh,
                                    ring.shm_fd.as_raw_fd(),
                                    ring.width,
                                    ring.height,
                                    true,
                                ) {
                                    Err(e) => {
                                        video_direct_backoff = VIDEO_DIRECT_RETRY_TICKS;
                                        warn!(
                                            "Video direct present not brought up: {}. Retrying in {} ticks; fallback: copy through present_frame",
                                            e, VIDEO_DIRECT_RETRY_TICKS
                                        );
                                    }
                                    Ok(()) => {
                                        info!(
                                            "Video zero-copy direct present activated ({}x{})",
                                            ring.width, ring.height
                                        );
                                    }
                                }
                            }
                        }
                    }

                    let next_slot = video_slot ^ 1;
                    if wl_ctx.direct_slot_free(next_slot) {
                        video_slot ^= 1;
                        let slot = video_slot as usize;

                        if let Some(slot_slice) = wl_ctx.direct_slot_slice(slot) {
                            match video.decode_next_frame(slot_slice) {
                                Ok(true) => match wl_ctx.present_shm_slot(slot) {
                                    Ok(true) => {
                                        first_frame_rendered = true;
                                        direct_path_used = true;
                                    }
                                    Ok(false) => {}
                                    Err(e) => {
                                        error!("Zero-copy present of video frame failed: {}", e)
                                    }
                                },
                                Ok(false) => {}
                                Err(e) => {
                                    direct_path_used = true;
                                    error!("Failed to decode frame into direct SHM: {}", e);
                                }
                            }
                        }
                    } else {
                        direct_path_used = true;
                    }
                }

                if !direct_path_used {
                    let needed = (video.width as usize) * (video.height as usize) * 4;
                    if frame_scratch.len() != needed {
                        frame_scratch.resize(needed, 0);
                    }
                    match video.decode_next_frame(&mut frame_scratch) {
                        Ok(true) => {
                            if let Some((_, _, ref mut wl_ctx)) = wayland_session {
                                if let Err(e) = wl_ctx.present_frame(&frame_scratch) {
                                    error!("Failed to present video frame on Wayland: {}", e);
                                }
                            }
                            first_frame_rendered = true;
                        }
                        Ok(false) => {}
                        Err(e) => error!("ffmpeg-next frame decode failed: {}", e),
                    }
                }
                video.target_fps
            }
            ActiveEngine::RustBin(runner) => {
                let fps = runner.target_fps;
                let runner_dims = (runner.width, runner.height);
                let direct_fd = runner.shm_raw_fd();

                if let Some((_, _, ref mut wl_ctx)) = wayland_session {
                    wl_ctx.reset_direct_if_stale(Some((direct_fd, runner_dims.0, runner_dims.1)));

                    if !wl_ctx.direct_slot_free(runner.peek_next_slot()) {
                        let frame_budget = Duration::from_secs_f64(1.0 / (fps.max(1) as f64));
                        let elapsed = frame_start.elapsed();
                        if elapsed < frame_budget {
                            std::thread::sleep(frame_budget - elapsed);
                        }
                        continue;
                    }
                }

                match runner.request_frame(niri_state) {
                    Ok(Some((slot, frame_bytes))) => {
                        if let Some((_, ref queue, ref mut wl_ctx)) = wayland_session {
                            if wl_ctx.direct.is_none()
                                && runner_dims == (wl_ctx.width, wl_ctx.height)
                            {
                                let qh = queue.handle();
                                if let Err(e) = wl_ctx.init_direct_shm(
                                    &qh,
                                    direct_fd,
                                    runner_dims.0,
                                    runner_dims.1,
                                    false,
                                ) {
                                    warn!(
                                        "Rust wallpaper direct present not brought up: {}. Fallback: copy through present_frame",
                                        e
                                    );
                                } else {
                                    info!(
                                        "Zero-copy direct present activated over Rust wallpaper memfd ({}x{})",
                                        runner_dims.0, runner_dims.1
                                    );
                                }
                            }

                            match wl_ctx.present_shm_slot(slot as usize) {
                                Ok(true) => {}
                                Ok(false) => {
                                    if let Err(e) = wl_ctx.present_frame(frame_bytes) {
                                        error!(
                                            "Failed to present Rust wallpaper frame on Wayland: {}",
                                            e
                                        );
                                    }
                                }
                                Err(e) => {
                                    error!(
                                        "Zero-copy present of Rust wallpaper frame failed: {}",
                                        e
                                    );
                                    if let Err(e) = wl_ctx.present_frame(frame_bytes) {
                                        error!(
                                            "Failed to present Rust wallpaper frame on Wayland: {}",
                                            e
                                        );
                                    }
                                }
                            }
                        }
                        first_frame_rendered = true;
                    }
                    Ok(None) => {}
                    Err(e) => error!("Failed to obtain a frame from Rust wallpaper: {}", e),
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
    info!("RWallpprs daemon shut down cleanly.");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_cli_client(
    mut stream: UnixStream,
    cfg: &DaemonConfig,
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
        } => match VideoWallpaperEngine::new(
            path.clone(),
            hw_accel,
            scale,
            fps_limit,
            width,
            height,
            cfg.video.drm_render_node.clone(),
            cfg.video.loop_playback,
        ) {
            Ok(engine) => {
                let codec = engine.codec;
                *active_engine = ActiveEngine::Video(engine);
                *first_frame_rendered = false;
                ControlResponse::Ok {
                    message: format!(
                        "Started native ffmpeg-next decoder for {:?} (codec {:?}, hw={:?}, {}x{}@{}fps){}",
                        path,
                        codec,
                        hw_accel,
                        width,
                        height,
                        fps_limit,
                        power_note(
                            power_mgr,
                            niri_monitor,
                            RustWallpaperMode::Animated,
                            *first_frame_rendered
                        )
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
                        "Started native Rust wallpaper {:?} (PID={}, mode={:?}, {}x{}@{}fps){}",
                        bin_path,
                        pid,
                        mode,
                        width,
                        height,
                        fps,
                        power_note(power_mgr, niri_monitor, mode, *first_frame_rendered)
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
                        "Embedded binary '{}' ({} KB) loaded into persistent daemon memfd and started (PID={}, mode={:?}){}",
                        name,
                        elf_payload.len() / 1024,
                        pid,
                        mode,
                        power_note(power_mgr, niri_monitor, mode, *first_frame_rendered)
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
                message: "Niri power saving and auto-hide policy updated".into(),
            }
        }
        ControlCommand::TogglePause => {
            power_mgr.user_paused = !power_mgr.user_paused;
            ControlResponse::Ok {
                message: format!("Pause toggled: user_paused={}", power_mgr.user_paused),
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
            let outputs = wayland_session
                .as_ref()
                .map(|(_, _, ctx)| ctx.query_outputs(surface_visible, current_fps))
                .unwrap_or_else(|| {
                    vec![OutputStatus {
                        name: "Niri-Primary-Output".into(),
                        width,
                        height,
                        refresh_mhz: 60_000,
                        surface_visible,
                        current_fps,
                    }]
                });

            ControlResponse::Status(DaemonStatusReport {
                pid: std::process::id(),
                niri_socket_connected: niri_monitor.is_connected(),
                active_source: source_desc,
                power_state,
                policy: power_mgr.policy.clone(),
                outputs,
                niri_state,
            })
        }
        ControlCommand::Shutdown => {
            *running = false;
            ControlResponse::Ok {
                message: "RWallpprs daemon is shutting down".into(),
            }
        }
    };

    write_framed_msg(&mut stream, &resp)?;
    Ok(())
}

fn log_power_transition(last_reason: &mut Option<String>, state: &PowerRuntimeState) {
    let signature = match state {
        PowerRuntimeState::ActiveRendering => None,
        PowerRuntimeState::UserPaused => Some("user_paused".to_string()),
        PowerRuntimeState::FrozenStatic { reason }
        | PowerRuntimeState::HiddenOccluded { reason } => Some(reason.clone()),
    };
    if *last_reason == signature {
        return;
    }
    match power_state_hint(state) {
        Some(hint) => warn!("Wallpaper paused: {hint}"),
        None => info!("Wallpaper resumed and is rendering"),
    }
    *last_reason = signature;
}

fn power_note(
    power_mgr: &mut PowerManager,
    niri_monitor: &NiriMonitor,
    mode: RustWallpaperMode,
    first_frame_rendered: bool,
) -> String {
    match power_state_hint(&power_mgr.evaluate(
        &niri_monitor.snapshot(),
        mode,
        first_frame_rendered,
    )) {
        Some(hint) => format!(" [paused: {hint}]"),
        None => String::new(),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_daemon(args.config)
}

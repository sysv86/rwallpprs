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
use power_manager::PowerManager;
use rust_runner::RustWallpaperRunner;
use rwallpprs_ipc::{
    read_framed_msg, write_framed_msg, ControlCommand, ControlResponse, DaemonStatusReport,
    OutputStatus, PowerRuntimeState, RustWallpaperMode, ShmBufferHeader,
};
use rustix::{
    fd::{AsRawFd, OwnedFd},
    fs::{ftruncate, memfd_create, MemfdFlags},
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

/// memfd демона под zero-copy direct present видео-движка: демон сам владеет
/// SHM-кольцом (ffmpeg-декодер пишет кадры сразу в память, отображаемую
/// композитором через `DirectShmPresent`).
struct VideoDirectRing {
    shm_fd: OwnedFd,
    width: u32,
    height: u32,
}

impl VideoDirectRing {
    fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        // CLOEXEC безопасен: кольцо никуда не передаётся (никаких exec-детей),
        // это SHM демона под wl_shm-пул композитора
        let shm_fd = memfd_create("rwallpprs-video-direct", MemfdFlags::CLOEXEC)
            .context("Ошибка rustix::fs::memfd_create для видео-SHM кольца")?;
        ftruncate(
            &shm_fd,
            ShmBufferHeader::new(width, height).total_shm_size() as u64,
        )
        .context("Ошибка rustix::fs::ftruncate для видео-SHM кольца")?;
        Ok(Self {
            shm_fd,
            width,
            height,
        })
    }

    /// Подходит ли кольцо под актуальное разрешение монитора
    fn fits(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }
}

pub fn control_socket_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(runtime_dir).join("rwallpprs.sock")
}

/// Минимум тактов главного цикла между повторными попытками подъёма
/// direct-презентера видео (ретрай с бэкоффом вместо шторма каждый кадр).
const VIDEO_DIRECT_RETRY_TICKS: u32 = 30;

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
    // Zero-copy direct present для видео: memfd-кольцо демона и чередуемый слот 0/1
    let mut video_direct_ring: Option<VideoDirectRing> = None;
    // Ретрай-бэкофф init direct-презентера видео: такты до следующей попытки
    let mut video_direct_backoff: u32 = 0;
    let mut video_slot: u8 = 0;
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
                // --- Zero-copy direct present: кадр пишется ffmpeg'ом сразу в memfd ---
                // `direct_path_used` => такт обработан direct-путём (кадр показан
                // или осознанно пропущен по flow-control): копирующий fallback не нужен
                let mut direct_path_used = false;

                if let Some((_, ref queue, ref mut wl_ctx)) = wayland_session {
                    // Согласованность direct-презентера с текущим видео-кольцом:
                    // после смены движка (SetVideo/SetRustBin/SetEmbeddedBin)
                    // запоротый презентер иначе продолжил бы показывать композитору
                    // буферы чужого memfd. Кольца нет => любой существующий
                    // direct чужой в принципе.
                    let ring_src = video_direct_ring
                        .as_ref()
                        .map(|ring| (ring.shm_fd.as_raw_fd(), ring.width, ring.height));
                    wl_ctx.reset_direct_if_stale(ring_src);

                    // Ленивый подъём кольца и direct-презентера, пересоздание после
                    // смены разрешения (configure сбрасывает direct). При ошибке —
                    // ретрай с бэкоффом: memfd-кольцо переиспользуется, выбрасывать
                    // его и штормить warn'ами каждый кадр нельзя.
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
                                        "Ошибка создания memfd видео-SHM демона: {}. Ретрай через {} тактов; фоллбэк: копирование через present_frame",
                                        e, VIDEO_DIRECT_RETRY_TICKS
                                    );
                                }
                            }
                        }
                        if video_direct_backoff == 0 {
                            if let Some(ring) = video_direct_ring.as_ref() {
                                let qh = queue.handle();
                                // write_header=true: у memfd демона заголовок
                                // появляется только здесь, при инициализации презентера
                                match wl_ctx.init_direct_shm(
                                    &qh,
                                    ring.shm_fd.as_raw_fd(),
                                    ring.width,
                                    ring.height,
                                    true,
                                ) {
                                    Err(e) => {
                                        // Кольцо живо (memfd переиспользуется),
                                        // сбрасываем только презентера и ждём бэкофф
                                        video_direct_backoff = VIDEO_DIRECT_RETRY_TICKS;
                                        warn!(
                                            "Direct present видео не поднят: {}. Ретрай через {} тактов; фоллбэк: копирование через present_frame",
                                            e, VIDEO_DIRECT_RETRY_TICKS
                                        );
                                    }
                                    Ok(()) => {
                                        info!(
                                            "Zero-copy direct present видео активирован ({}x{})",
                                            ring.width, ring.height
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Flow-control ДО декодирования: если следующий слот кольца ещё
                    // в композиторе (release не пришёл), декодировать кадр некуда —
                    // честный drop: не тратим кадр ffmpeg и не перезаписываем
                    // память буфера, который композитор ещё мог бы читать.
                    let next_slot = video_slot ^ 1;
                    if wl_ctx.direct_slot_free(next_slot) {
                        // Чередуем слоты 0/1 (аналог current_slot у Rust-раннера)
                        video_slot ^= 1;
                        let slot = video_slot as usize;

                        // Декодирование ffmpeg'ом прямо в слот memfd — без промежуточных буферов
                        if let Some(slot_slice) = wl_ctx.direct_slot_slice(slot) {
                            match video.decode_next_frame(slot_slice) {
                                Ok(true) => match wl_ctx.present_shm_slot(slot) {
                                    Ok(true) => {
                                        first_frame_rendered = true;
                                        direct_path_used = true;
                                    }
                                    Ok(false) => {
                                        // Честный drop: кадр уже записан в слот кольца,
                                        // attach не прошёл — повторить его нельзя,
                                        // восстановление на следующей итерации невозможно
                                    }
                                    Err(e) => error!("Ошибка zero-copy present видеокадра: {}", e),
                                },
                                Ok(false) => {}
                                Err(e) => {
                                    // Кадр при ошибке direct-декода не пересоздаём
                                    // фолбэком (иначе декодер гоняется дважды);
                                    // следующий такт попробует снова.
                                    direct_path_used = true;
                                    error!("Ошибка декодирования кадра в direct-SHM: {}", e);
                                }
                            }
                        }
                    } else {
                        // Слот занят: такт пропущен без декодирования — кадр
                        // ffmpeg'ом не съедается почём зря, fallback не запускается
                        direct_path_used = true;
                    }
                }

                // --- Фоллбэк: старый путь со scratch-буфером и копией в SlotPool ---
                if !direct_path_used {
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
                }
                video.target_fps
            }
            ActiveEngine::RustBin(runner) => {
                let fps = runner.target_fps;
                // Параметры memfd раннера фиксируются на момент спавна: снимаем их
                // до позаимствованного request_frame (срез кадра держит мут-заём)
                let runner_dims = (runner.width, runner.height);
                let direct_fd = runner.shm_raw_fd();

                // Согласованность direct с текущим memfd runner'а: после смены
                // движка запоротый презентер (чужой memfd, возможно даже под
                // переиспользованным номером fd — identity проверяется) сбрасывается
                if let Some((_, _, ref mut wl_ctx)) = wayland_session {
                    wl_ctx.reset_direct_if_stale(Some((direct_fd, runner_dims.0, runner_dims.1)));

                    // Flow-control: слот, который runner займёт следующим кадром,
                    // должен быть уже отозван композитором. Иначе RenderFrame
                    // перезапишет буфер, который композитор ещё читает. current_slot
                    // runner'а НЕ двигается — request_frame на этом такте не зовётся.
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
                            // Zero-copy direct present поверх memfd самого Rust-обоя:
                            // поднимается лениво на первом FrameReady
                            if wl_ctx.direct.is_none() && runner_dims == (wl_ctx.width, wl_ctx.height)
                            {
                                let qh = queue.handle();
                                // write_header=false: заголовок memfd уже записан
                                // spawn_internal при запуске runner'а
                                if let Err(e) = wl_ctx.init_direct_shm(
                                    &qh, direct_fd, runner_dims.0, runner_dims.1, false,
                                ) {
                                    warn!(
                                        "Direct present Rust-обоя не поднят: {}. Фоллбэк: копирование через present_frame",
                                        e
                                    );
                                } else {
                                    info!(
                                        "Zero-copy direct present активирован поверх memfd Rust-обоя ({}x{})",
                                        runner_dims.0, runner_dims.1
                                    );
                                }
                            }

                            // Прямой attach слота, куда движок только что отрендерил кадр
                            match wl_ctx.present_shm_slot(slot as usize) {
                                Ok(true) => {}
                                Ok(false) => {
                                    // Слот занят сервером (release не пришёл): фоллбэк
                                    // на копирующий путь, чтобы кадр не потерять
                                    if let Err(e) = wl_ctx.present_frame(frame_bytes) {
                                        error!("Ошибка вывода кадра Rust-обоев на Wayland: {}", e);
                                    }
                                }
                                Err(e) => {
                                    error!("Ошибка zero-copy present кадра Rust-обоев: {}", e);
                                    // Фоллбэк: старый копирующий путь через SlotPool
                                    if let Err(e) = wl_ctx.present_frame(frame_bytes) {
                                        error!("Ошибка вывода кадра Rust-обоев на Wayland: {}", e);
                                    }
                                }
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

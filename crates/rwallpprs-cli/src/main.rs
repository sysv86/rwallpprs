//! CLI-утилита `rwallpprs` (`v0.2.0-beta.1`).
//!
//! При запуске `rwallpprs run-embedded` извлекает вложенный ELF-образ обоев из конца
//! собственного файла и передает его по фреймированному сокету в `rwallpprs-daemon`
//! (`ControlCommand::SetEmbeddedBin`), где демон сохраняет его в персистентный `memfd`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rwallpprs_ipc::{
    read_framed_msg, write_framed_msg, ControlCommand, ControlResponse, HwAccelBackend,
    PowerSavePolicy, RustWallpaperMode, ScaleMode, EMBEDDED_BIN_MAGIC,
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    name = "rwallpprs",
    version,
    about = "Видео (ffmpeg-next H.264/HEVC/AV1/VP9) и Нативные GPU/CPU Rust-обои для композитора Niri (Arch Linux)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliHwAccel {
    Auto,
    Vaapi,
    Vulkan,
    Nvdec,
    Vdpau,
    Software,
}

impl From<CliHwAccel> for HwAccelBackend {
    fn from(v: CliHwAccel) -> Self {
        match v {
            CliHwAccel::Auto => Self::Auto,
            CliHwAccel::Vaapi => Self::Vaapi,
            CliHwAccel::Vulkan => Self::Vulkan,
            CliHwAccel::Nvdec => Self::Nvdec,
            CliHwAccel::Vdpau => Self::Vdpau,
            CliHwAccel::Software => Self::Software,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliScaleMode {
    Fill,
    Fit,
    Stretch,
}

impl From<CliScaleMode> for ScaleMode {
    fn from(v: CliScaleMode) -> Self {
        match v {
            CliScaleMode::Fill => Self::Fill,
            CliScaleMode::Fit => Self::Fit,
            CliScaleMode::Stretch => Self::Stretch,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliRustMode {
    Animated,
    Static,
    Reactive,
}

impl From<CliRustMode> for RustWallpaperMode {
    fn from(v: CliRustMode) -> Self {
        match v {
            CliRustMode::Animated => Self::Animated,
            CliRustMode::Static => Self::Static,
            CliRustMode::Reactive => Self::ReactiveNiri,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ToggleSwitch {
    On,
    Off,
}

impl ToggleSwitch {
    fn as_bool(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Запустить фоновый демон RWallpprs под Niri
    Daemon {
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Установить видео-обои через нативный декодер ffmpeg-next (H.264, HEVC, VP9, AV1)
    SetVideo {
        path: PathBuf,
        #[arg(short, long)]
        output: Option<String>,
        #[arg(long, value_enum, default_value_t = CliHwAccel::Auto)]
        hw_accel: CliHwAccel,
        #[arg(long, default_value_t = 60)]
        fps: u32,
        #[arg(long, value_enum, default_value_t = CliScaleMode::Fill)]
        scale: CliScaleMode,
    },

    /// Запустить скомпилированный бинарник Rust-обоев на рабочем столе Niri
    SetBin {
        bin_path: PathBuf,
        #[arg(short, long, value_enum, default_value_t = CliRustMode::Animated)]
        mode: CliRustMode,
        #[arg(long, default_value_t = 60)]
        fps: u32,
        #[arg(short, long)]
        output: Option<String>,
        #[arg(last = true)]
        args: Vec<String>,
    },

    /// Вшить бинарник Rust-обоев внутрь исполняемого файла CLI
    BundleBin {
        bin_path: PathBuf,
        #[arg(short, long)]
        output_cli: PathBuf,
    },

    /// Передать вшитый в этот CLI бинарник обоев в персистентный memfd демона и запустить
    RunEmbedded {
        #[arg(short, long, value_enum, default_value_t = CliRustMode::Animated)]
        mode: CliRustMode,
        #[arg(long, default_value_t = 60)]
        fps: u32,
        #[arg(last = true)]
        args: Vec<String>,
    },

    /// Настройка энергосбережения и автоматического скрытия обоев в Niri
    Power {
        #[arg(long, value_enum)]
        enabled: Option<ToggleSwitch>,
        #[arg(long, value_enum)]
        auto_hide: Option<ToggleSwitch>,
        #[arg(long, value_enum)]
        pause_fullscreen: Option<ToggleSwitch>,
        #[arg(long, value_enum)]
        pause_maximized: Option<ToggleSwitch>,
        #[arg(long, value_enum)]
        wake_overview: Option<ToggleSwitch>,
        #[arg(long)]
        battery_limit: Option<u8>,
        #[arg(long)]
        toggle_pause: bool,
    },

    /// Показать текущий статус демона, кодек/PID и состояние окон Niri
    Status,

    /// Остановить демон RWallpprs
    Stop,
}

fn control_socket_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(runtime_dir).join("rwallpprs.sock")
}

fn send_command(cmd: &ControlCommand) -> Result<ControlResponse> {
    let sock = control_socket_path();
    let mut stream = UnixStream::connect(&sock).with_context(|| {
        format!(
            "Не удалось подключиться к демону RWallpprs по сокету {:?}. Запустите `rwallpprs daemon`",
            sock
        )
    })?;
    write_framed_msg(&mut stream, cmd)?;
    let resp: ControlResponse = read_framed_msg(&mut stream)?;
    Ok(resp)
}

fn bundle_binary_into_cli(wallpaper_bin: &PathBuf, out_cli: &PathBuf) -> Result<()> {
    let self_exe = std::env::current_exe().context("Не удалось определить текущий бинарник CLI")?;
    let mut cli_bytes = std::fs::read(&self_exe)?;
    let wp_bytes = std::fs::read(wallpaper_bin)
        .with_context(|| format!("Не удалось прочитать бинарник обоев {:?}", wallpaper_bin))?;

    let payload_len = wp_bytes.len() as u64;
    cli_bytes.extend_from_slice(&wp_bytes);
    cli_bytes.extend_from_slice(&payload_len.to_le_bytes());
    cli_bytes.extend_from_slice(EMBEDDED_BIN_MAGIC);

    std::fs::write(out_cli, &cli_bytes)?;
    let mut perms = std::fs::metadata(out_cli)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(out_cli, perms)?;

    println!(
        "Бинарник обоев {:?} ({} КБ) вложен в {:?}.
Запуск: {:?} run-embedded --mode animated --fps 60",
        wallpaper_bin,
        wp_bytes.len() / 1024,
        out_cli,
        out_cli
    );
    Ok(())
}

fn read_embedded_elf_payload() -> Result<(String, Vec<u8>)> {
    let self_exe = std::env::current_exe()?;
    let exe_name = self_exe
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "embedded-wallpaper".into());

    let mut file = File::open(&self_exe)?;
    let total_len = file.metadata()?.len();

    let footer_len = (EMBEDDED_BIN_MAGIC.len() + 8) as u64;
    if total_len <= footer_len {
        anyhow::bail!("В данный CLI-файл не вложен бинарник обоев");
    }

    file.seek(SeekFrom::End(-(footer_len as i64)))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let payload_len = u64::from_le_bytes(len_buf);

    let mut magic_buf = [0u8; 18];
    file.read_exact(&mut magic_buf)?;
    if &magic_buf != EMBEDDED_BIN_MAGIC {
        anyhow::bail!(
            "Сигнатура RWALLPPRS_EMBED_V2 не найдена. Сначала упакуйте бинарник командой `rwallpprs bundle-bin <BIN> -o <OUT>`"
        );
    }

    let payload_offset = total_len
        .checked_sub(footer_len + payload_len)
        .context("Поврежденный размер вложенного бинарника")?;
    file.seek(SeekFrom::Start(payload_offset))?;

    let mut payload = vec![0u8; payload_len as usize];
    file.read_exact(&mut payload)?;
    Ok((exe_name, payload))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Daemon { config } => {
            let mut cmd = Command::new("rwallpprs-daemon");
            if let Some(cfg) = config {
                cmd.arg("--config").arg(cfg);
            }
            let status = cmd
                .status()
                .context("Не удалось запустить `rwallpprs-daemon`")?;
            if !status.success() {
                anyhow::bail!("rwallpprs-daemon завершился с кодом {:?}", status.code());
            }
        }
        Commands::SetVideo {
            path,
            output,
            hw_accel,
            fps,
            scale,
        } => {
            let abs_path = std::fs::canonicalize(&path).unwrap_or(path);
            let resp = send_command(&ControlCommand::SetVideo {
                path: abs_path,
                output,
                hw_accel: hw_accel.into(),
                fps_limit: fps,
                scale: scale.into(),
            })?;
            print_response(resp);
        }
        Commands::SetBin {
            bin_path,
            mode,
            fps,
            output,
            args,
        } => {
            let abs_path = std::fs::canonicalize(&bin_path).unwrap_or(bin_path);
            let resp = send_command(&ControlCommand::SetRustBin {
                bin_path: abs_path,
                args,
                output,
                mode: mode.into(),
                fps,
            })?;
            print_response(resp);
        }
        Commands::BundleBin {
            bin_path,
            output_cli,
        } => {
            bundle_binary_into_cli(&bin_path, &output_cli)?;
        }
        Commands::RunEmbedded { mode, fps, args } => {
            let (name, elf_payload) = read_embedded_elf_payload()?;
            let resp = send_command(&ControlCommand::SetEmbeddedBin {
                name,
                elf_payload,
                args,
                output: None,
                mode: mode.into(),
                fps,
            })?;
            print_response(resp);
        }
        Commands::Power {
            enabled,
            auto_hide,
            pause_fullscreen,
            pause_maximized,
            wake_overview,
            battery_limit,
            toggle_pause,
        } => {
            if toggle_pause {
                let resp = send_command(&ControlCommand::TogglePause)?;
                print_response(resp);
                return Ok(());
            }

            let mut policy = match send_command(&ControlCommand::GetStatus)? {
                ControlResponse::Status(st) => st.policy,
                _ => PowerSavePolicy::default(),
            };

            if let Some(v) = enabled {
                policy.enabled = v.as_bool();
            }
            if let Some(v) = auto_hide {
                policy.auto_hide_surface = v.as_bool();
            }
            if let Some(v) = pause_fullscreen {
                policy.pause_on_fullscreen = v.as_bool();
            }
            if let Some(v) = pause_maximized {
                policy.pause_on_maximized_column = v.as_bool();
            }
            if let Some(v) = wake_overview {
                policy.wake_on_niri_overview = v.as_bool();
            }
            if let Some(limit) = battery_limit {
                policy.battery_freeze_threshold = limit.min(100);
            }

            let resp = send_command(&ControlCommand::UpdatePowerPolicy(policy))?;
            print_response(resp);
        }
        Commands::Status => {
            let resp = send_command(&ControlCommand::GetStatus)?;
            print_response(resp);
        }
        Commands::Stop => {
            let resp = send_command(&ControlCommand::Shutdown)?;
            print_response(resp);
        }
    }

    Ok(())
}

fn print_response(resp: ControlResponse) {
    match resp {
        ControlResponse::Ok { message } => println!("[OK] {}", message),
        ControlResponse::Error { reason } => eprintln!("[ОШИБКА] {}", reason),
        ControlResponse::Status(st) => {
            println!("=== RWallpprs v0.2.0-beta.1 Daemon Status (Niri) ===");
            println!("PID демона          : {}", st.pid);
            println!("NIRI_SOCKET активен : {}", st.niri_socket_connected);
            println!("Активный источник   : {}", st.active_source);
            println!("Состояние рендера   : {:?}", st.power_state);
            println!(
                "Niri Viewport       : workspace=#{} (id={}), overview={}, fullscreen={}, focused_ratio={:.2}, screen_coverage={:.2}, scroll={:.1}",
                st.niri_state.active_workspace_idx,
                st.niri_state.active_workspace_id,
                st.niri_state.overview_open,
                st.niri_state.has_fullscreen_window,
                st.niri_state.focused_window_width_ratio,
                st.niri_state.screen_coverage_ratio,
                st.niri_state.strip_scroll_offset
            );
            println!(
                "Энергосбережение    : enabled={}, auto_hide={}, fullscreen_pause={}, max_col_pause={} (>={:.0}%), bat_limit={}%",
                st.policy.enabled,
                st.policy.auto_hide_surface,
                st.policy.pause_on_fullscreen,
                st.policy.pause_on_maximized_column,
                st.policy.maximized_column_threshold * 100.0,
                st.policy.battery_freeze_threshold
            );
            for out in st.outputs {
                println!(
                    "Монитор {:<12}: {}x{} | wl_surface видима={} | целевой FPS={:.0}",
                    out.name, out.width, out.height, out.surface_visible, out.current_fps
                );
            }
        }
    }
}

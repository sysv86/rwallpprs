//! Супервизор скомпилированных бинарников Rust-обоев (`RustWallpaperRunner`).
//!
//! Улучшения `v0.2.0-beta.1`:
//! 1. Использует безопасные вызовы [`rustix::fs::memfd_create`], [`rustix::fs::ftruncate`] и [`rustix::mm::mmap`].
//! 2. Поддерживает хранение **вложенного ELF-образа (`embedded_elf_fd: Option<OwnedFd>`)** прямо внутри демона!
//!    Когда пользователь вызывает `rwallpprs run-embedded`, демон сам создает долгоживущий `memfd` с ELF-кодом
//!    обоев и запускает `/proc/self/fd/<fd>`, поэтому CLI может завершиться мгновенно, а демон способен
//!    перезапускать вложенный бинарник в любой момент.

use anyhow::{Context, Result};
use rustix::{
    fd::{AsRawFd, OwnedFd, RawFd},
    fs::{fcntl_getfd, fcntl_setfd, ftruncate, memfd_create, FdFlags, MemfdFlags},
    mm::{mmap, munmap, MapFlags, ProtFlags},
};
use rwallpprs_ipc::{
    NiriViewportState, RustWallpaperMode, ShmBufferHeader, WorkerAck, WorkerEvent,
};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::ptr::NonNull;
use std::time::Instant;
use tracing::{info, warn};

pub struct RustWallpaperRunner {
    pub display_name: String,
    pub exec_path: PathBuf,
    pub args: Vec<String>,
    pub mode: RustWallpaperMode,
    pub target_fps: u32,
    pub width: u32,
    pub height: u32,
    shm_fd: OwnedFd,
    shm_ptr: NonNull<u8>,
    shm_size: usize,
    /// Если обои запущены через `run-embedded`, этот `OwnedFd` удерживает их ELF-образ в RAM демона.
    _embedded_elf_fd: Option<OwnedFd>,
    child: Child,
    child_stdin: ChildStdin,
    child_stdout: BufReader<ChildStdout>,
    current_slot: u8,
    frame_index: u64,
    start_instant: Instant,
    last_frame_instant: Instant,
    suspended: bool,
    last_niri_snapshot: Option<NiriViewportState>,
}

unsafe impl Send for RustWallpaperRunner {}

impl RustWallpaperRunner {
    /// Запускает внешний скомпилированный бинарник Rust-обоев по пути на диске.
    pub fn spawn_from_path(
        bin_path: PathBuf,
        args: Vec<String>,
        mode: RustWallpaperMode,
        target_fps: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        if !bin_path.exists() {
            anyhow::bail!("Бинарник Rust-обоев не найден: {:?}", bin_path);
        }
        let display_name = bin_path.display().to_string();
        Self::spawn_internal(display_name, bin_path, None, args, mode, target_fps, width, height)
    }

    /// Запускает вложенный в CLI ELF-бинарник обоев напрямую из памяти демона (`memfd_create`).
    pub fn spawn_from_embedded_elf(
        name: String,
        elf_payload: &[u8],
        args: Vec<String>,
        mode: RustWallpaperMode,
        target_fps: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // Создаем долгоживущий memfd в адресном пространстве демона с флагом CLOEXEC
        let elf_fd = memfd_create("rwallpprs-embedded-elf", MemfdFlags::CLOEXEC)
            .context("Не удалось создать memfd для вложенного ELF-образа обоев")?;

        let mut elf_file = std::fs::File::from(elf_fd);
        elf_file
            .write_all(elf_payload)
            .context("Не удалось записать ELF-образ в memfd демона")?;
        elf_file.flush()?;
        let elf_fd: OwnedFd = elf_file.into();

        let exec_path = PathBuf::from(format!("/proc/self/fd/{}", elf_fd.as_raw_fd()));
        let display_name = format!("embedded:{} ({} КБ в RAM)", name, elf_payload.len() / 1024);

        Self::spawn_internal(
            display_name,
            exec_path,
            Some(elf_fd),
            args,
            mode,
            target_fps,
            width,
            height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_internal(
        display_name: String,
        exec_path: PathBuf,
        embedded_elf_fd: Option<OwnedFd>,
        args: Vec<String>,
        mode: RustWallpaperMode,
        target_fps: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let header = ShmBufferHeader::new(width, height);
        let shm_size = header.total_shm_size();

        // Создаем буфер кадров в разделяемой памяти без CLOEXEC, чтобы дочерний процесс унаследовал FD
        let shm_fd = memfd_create("rwallpprs-shm-framebuf", MemfdFlags::empty())
            .context("Ошибка вызова rustix::fs::memfd_create для кадрового буфера")?;

        ftruncate(&shm_fd, shm_size as u64)
            .context("Ошибка rustix::fs::ftruncate для SHM буфера")?;

        // Сбрасываем FD_CLOEXEC явно для передаваемого SHM дескриптора
        let flags = fcntl_getfd(&shm_fd)?;
        fcntl_setfd(&shm_fd, flags.difference(FdFlags::CLOEXEC))?;

        let raw_ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                shm_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &shm_fd,
                0,
            )
            .context("Ошибка rustix::mm::mmap в демоне")?
        };
        let shm_ptr = NonNull::new(raw_ptr.cast::<u8>())
            .ok_or_else(|| anyhow::anyhow!("mmap вернул нулевой указатель"))?;

        unsafe {
            std::ptr::write(shm_ptr.as_ptr().cast::<ShmBufferHeader>(), header);
        }

        let mut cmd = Command::new(&exec_path);
        cmd.args(&args)
            .env("RWALLPPRS_SHM_FD", shm_fd.as_raw_fd().to_string())
            .env("RWALLPPRS_SHM_SIZE", shm_size.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Не удалось запустить Rust-обои {}", display_name))?;

        let child_stdin = child
            .stdin
            .take()
            .context("Не удалось получить stdin дочернего процесса")?;
        let child_stdout = BufReader::new(
            child
                .stdout
                .take()
                .context("Не удалось получить stdout дочернего процесса")?,
        );

        info!(
            "Запущены Rust-обои {} (PID={}, mode={:?}, SHM={} КБ)",
            display_name,
            child.id(),
            mode,
            shm_size / 1024
        );

        let now = Instant::now();
        Ok(Self {
            display_name,
            exec_path,
            args,
            mode,
            target_fps: target_fps.max(1),
            width,
            height,
            shm_fd,
            shm_ptr,
            shm_size,
            _embedded_elf_fd: embedded_elf_fd,
            child,
            child_stdin,
            child_stdout,
            current_slot: 0,
            frame_index: 0,
            start_instant: now,
            last_frame_instant: now,
            suspended: false,
            last_niri_snapshot: None,
        })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Raw FD `memfd` кадрового буфера движка (zero-copy direct present поверх
    /// того же файла: демон импортирует его в `wl_shm` без копий пикселей).
    /// Runner всегда владеет этим memfd, поэтому `RawFd` без `Option`.
    pub fn shm_raw_fd(&self) -> RawFd {
        self.shm_fd.as_raw_fd()
    }

    /// Слот, который будет использован для **следующего** кадра — без продвижения
    /// `current_slot`. Нужен flow-control'у главного цикла: перед отправкой
    /// `RenderFrame` проверяется, освободил ли композитор этот слот.
    pub fn peek_next_slot(&self) -> u8 {
        self.current_slot
    }

    pub fn should_render_reactive(&mut self, niri: &NiriViewportState) -> bool {
        if self.frame_index == 0 {
            self.last_niri_snapshot = Some(niri.clone());
            return true;
        }
        if let Some(prev) = &self.last_niri_snapshot {
            let changed = prev.active_workspace_id != niri.active_workspace_id
                || prev.overview_open != niri.overview_open
                || (prev.strip_scroll_offset - niri.strip_scroll_offset).abs() > 0.5;
            if changed {
                self.last_niri_snapshot = Some(niri.clone());
                return true;
            }
            false
        } else {
            self.last_niri_snapshot = Some(niri.clone());
            true
        }
    }

    pub fn request_frame(&mut self, niri: NiriViewportState) -> Result<Option<(u8, &[u8])>> {
        if self.suspended {
            return Ok(None);
        }
        if self.mode == RustWallpaperMode::ReactiveNiri && !self.should_render_reactive(&niri) {
            return Ok(None);
        }

        let now = Instant::now();
        let time_secs = now.duration_since(self.start_instant).as_secs_f64();
        let dt_secs = now.duration_since(self.last_frame_instant).as_secs_f32();
        self.last_frame_instant = now;

        let slot = self.current_slot;
        self.current_slot ^= 1;

        let req = WorkerEvent::RenderFrame {
            slot,
            frame_index: self.frame_index,
            time_secs,
            dt_secs,
            width: self.width,
            height: self.height,
            niri,
        };
        self.frame_index += 1;

        serde_json::to_writer(&mut self.child_stdin, &req)?;
        self.child_stdin.write_all(b"
")?;
        self.child_stdin.flush()?;

        let mut line = String::new();
        self.child_stdout.read_line(&mut line)?;
        if line.trim().is_empty() {
            anyhow::bail!("Процесс Rust-обоев закрыл канал ответа");
        }

        let ack: WorkerAck = serde_json::from_str(&line)?;
        match ack {
            WorkerAck::FrameReady { slot, .. } => {
                let header = ShmBufferHeader::new(self.width, self.height);
                let offset = header.slot_offset(slot);
                let byte_len = header.slot_size_bytes as usize;
                let frame_slice = unsafe {
                    std::slice::from_raw_parts(self.shm_ptr.as_ptr().add(offset), byte_len)
                };
                // Возвращаем слот вместе со срезом: direct-present привязывает
                // wl_buffer к этому же офсету (нулевое копирование через SHM)
                Ok(Some((slot, frame_slice)))
            }
            WorkerAck::FrameSkipped { .. } => Ok(None),
        }
    }

    pub fn set_suspended(&mut self, suspend: bool) {
        if self.suspended == suspend {
            return;
        }
        self.suspended = suspend;
        let evt = if suspend {
            WorkerEvent::Suspend
        } else {
            WorkerEvent::Resume
        };
        if serde_json::to_writer(&mut self.child_stdin, &evt).is_ok() {
            let _ = self.child_stdin.write_all(b"
");
            let _ = self.child_stdin.flush();
        }
    }
}

impl Drop for RustWallpaperRunner {
    fn drop(&mut self) {
        let _ = serde_json::to_writer(&mut self.child_stdin, &WorkerEvent::Terminate);
        let _ = self.child_stdin.write_all(b"
");
        let _ = self.child_stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
        unsafe {
            let _ = munmap(self.shm_ptr.as_ptr().cast(), self.shm_size);
        }
        warn!(
            "Процесс Rust-обоев {} остановлен, ресурсы SHM освобождены",
            self.display_name
        );
    }
}

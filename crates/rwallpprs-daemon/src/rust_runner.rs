//! Supervisor for compiled Rust wallpaper binaries (`RustWallpaperRunner`).
//!
//! Responsibilities:
//! 1. Manages POSIX shared memory (`memfd_create`, `ftruncate`, `mmap`) via safe [`rustix`] wrappers.
//! 2. Preserves embedded ELF images (`embedded_elf_fd: Option<OwnedFd>`) inside the daemon across
//!    the entire session so `run-embedded` wallpapers can be cleanly restarted on monitor resolution
//!    changes (`resize_output`) or after unexpected worker crashes.
//! 3. Provides automatic crash recovery with backoff when a worker process terminates unexpectedly.

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
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::ptr::NonNull;
use std::time::{Duration, Instant};
use tracing::{info, warn};

const MAX_CRASH_RESTARTS: u32 = 3;
const CRASH_RESTART_COOLDOWN: Duration = Duration::from_secs(2);

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
    /// Keeps the ELF image of a wallpaper started via `run-embedded` alive in daemon RAM
    /// so `/proc/self/fd/<fd>` remains executable across resolution changes and restarts.
    embedded_elf_fd: Option<OwnedFd>,
    child: Child,
    child_stdin: ChildStdin,
    child_stdout: BufReader<ChildStdout>,
    current_slot: u8,
    frame_index: u64,
    start_instant: Instant,
    last_frame_instant: Instant,
    suspended: bool,
    last_niri_snapshot: Option<NiriViewportState>,
    crash_restarts: u32,
    last_restart_instant: Option<Instant>,
}

unsafe impl Send for RustWallpaperRunner {}

impl RustWallpaperRunner {
    /// Launches an external compiled Rust wallpaper binary from disk.
    pub fn spawn_from_path(
        bin_path: PathBuf,
        args: Vec<String>,
        mode: RustWallpaperMode,
        target_fps: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        if !bin_path.exists() {
            anyhow::bail!("Rust wallpaper binary not found: {:?}", bin_path);
        }
        let display_name = bin_path.display().to_string();
        Self::spawn_internal(
            display_name,
            bin_path,
            None,
            args,
            mode,
            target_fps,
            width,
            height,
        )
    }

    /// Launches a wallpaper ELF binary embedded in the CLI straight from daemon memory (`memfd_create`).
    pub fn spawn_from_embedded_elf(
        name: String,
        elf_payload: &[u8],
        args: Vec<String>,
        mode: RustWallpaperMode,
        target_fps: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let elf_fd = memfd_create("rwallpprs-embedded-elf", MemfdFlags::CLOEXEC)
            .context("Failed to create a memfd for the embedded wallpaper ELF image")?;

        let mut elf_file = std::fs::File::from(elf_fd);
        elf_file
            .write_all(elf_payload)
            .context("Failed to write the ELF image into the daemon memfd")?;
        elf_file.flush()?;
        let elf_fd: OwnedFd = elf_file.into();

        let exec_path = PathBuf::from(format!("/proc/self/fd/{}", elf_fd.as_raw_fd()));
        let display_name = format!("embedded:{} ({} KB in RAM)", name, elf_payload.len() / 1024);

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

    fn allocate_shm_and_spawn_child(
        display_name: &str,
        exec_path: &Path,
        args: &[String],
        width: u32,
        height: u32,
    ) -> Result<(
        OwnedFd,
        NonNull<u8>,
        usize,
        Child,
        ChildStdin,
        BufReader<ChildStdout>,
    )> {
        let header = ShmBufferHeader::new(width, height);
        let shm_size = header.total_shm_size();

        // Create shared memory frame buffer without CLOEXEC so the child inherits the FD.
        let shm_fd = memfd_create("rwallpprs-shm-framebuf", MemfdFlags::empty())
            .context("rustix::fs::memfd_create failed for the frame buffer")?;

        ftruncate(&shm_fd, shm_size as u64)
            .context("rustix::fs::ftruncate failed for the SHM buffer")?;

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
            .context("rustix::mm::mmap failed in the daemon")?
        };
        let shm_ptr = NonNull::new(raw_ptr.cast::<u8>())
            .ok_or_else(|| anyhow::anyhow!("mmap returned a null pointer"))?;

        unsafe {
            std::ptr::write(shm_ptr.as_ptr().cast::<ShmBufferHeader>(), header);
        }

        let mut cmd = Command::new(exec_path);
        cmd.args(args)
            .env("RWALLPPRS_SHM_FD", shm_fd.as_raw_fd().to_string())
            .env("RWALLPPRS_SHM_SIZE", shm_size.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                unsafe {
                    let _ = munmap(shm_ptr.as_ptr().cast(), shm_size);
                }
                return Err(e)
                    .with_context(|| format!("Failed to launch Rust wallpaper {}", display_name));
            }
        };

        let child_stdin = child
            .stdin
            .take()
            .context("Failed to capture child process stdin")?;
        let child_stdout = BufReader::new(
            child
                .stdout
                .take()
                .context("Failed to capture child process stdout")?,
        );

        Ok((shm_fd, shm_ptr, shm_size, child, child_stdin, child_stdout))
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
        let (shm_fd, shm_ptr, shm_size, child, child_stdin, child_stdout) =
            Self::allocate_shm_and_spawn_child(&display_name, &exec_path, &args, width, height)?;

        info!(
            "Started Rust wallpaper {} (PID={}, mode={:?}, {}x{}, SHM={} KB)",
            display_name,
            child.id(),
            mode,
            width,
            height,
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
            embedded_elf_fd,
            child,
            child_stdin,
            child_stdout,
            current_slot: 0,
            frame_index: 0,
            start_instant: now,
            last_frame_instant: now,
            suspended: false,
            last_niri_snapshot: None,
            crash_restarts: 0,
            last_restart_instant: None,
        })
    }

    /// Recreates the shared memory double buffer and restarts the wallpaper worker
    /// for a new monitor resolution while preserving `embedded_elf_fd`.
    pub fn resize_output(&mut self, new_width: u32, new_height: u32) -> Result<()> {
        if self.width == new_width && self.height == new_height {
            return Ok(());
        }
        info!(
            "Resizing Rust wallpaper {} from {}x{} to {}x{}",
            self.display_name, self.width, self.height, new_width, new_height
        );
        self.respawn_with_dimensions(new_width, new_height)?;
        self.crash_restarts = 0;
        Ok(())
    }

    fn respawn_with_dimensions(&mut self, new_width: u32, new_height: u32) -> Result<()> {
        // Refresh exec_path in case embedded_elf_fd is active.
        if let Some(ref elf_fd) = self.embedded_elf_fd {
            self.exec_path = PathBuf::from(format!("/proc/self/fd/{}", elf_fd.as_raw_fd()));
        }

        let (new_shm_fd, new_shm_ptr, new_shm_size, new_child, new_stdin, new_stdout) =
            Self::allocate_shm_and_spawn_child(
                &self.display_name,
                &self.exec_path,
                &self.args,
                new_width,
                new_height,
            )?;

        // Terminate the previous child and release the old SHM mapping.
        let _ = serde_json::to_writer(&mut self.child_stdin, &WorkerEvent::Terminate);
        let _ = self.child_stdin.write_all(b"\n");
        let _ = self.child_stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
        unsafe {
            let _ = munmap(self.shm_ptr.as_ptr().cast(), self.shm_size);
        }

        self.shm_fd = new_shm_fd;
        self.shm_ptr = new_shm_ptr;
        self.shm_size = new_shm_size;
        self.child = new_child;
        self.child_stdin = new_stdin;
        self.child_stdout = new_stdout;
        self.width = new_width;
        self.height = new_height;
        self.current_slot = 0;
        self.frame_index = 0;
        self.last_frame_instant = Instant::now();
        self.last_niri_snapshot = None;
        self.suspended = false;

        Ok(())
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Raw FD of the worker frame-buffer `memfd` for zero-copy `wl_shm` presentation.
    pub fn shm_raw_fd(&self) -> RawFd {
        self.shm_fd.as_raw_fd()
    }

    /// Slot index that will be used for the next frame without advancing `current_slot`.
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

        let io_result = (|| -> Result<String> {
            serde_json::to_writer(&mut self.child_stdin, &req)?;
            self.child_stdin.write_all(b"\n")?;
            self.child_stdin.flush()?;

            let mut line = String::new();
            self.child_stdout.read_line(&mut line)?;
            if line.trim().is_empty() {
                anyhow::bail!("Rust wallpaper worker closed its reply pipe");
            }
            Ok(line)
        })();

        let line = match io_result {
            Ok(l) => l,
            Err(err) => {
                return self.handle_worker_crash(err);
            }
        };

        let ack: WorkerAck = serde_json::from_str(&line)?;
        match ack {
            WorkerAck::FrameReady { slot, .. } => {
                let header = ShmBufferHeader::new(self.width, self.height);
                let offset = header.slot_offset(slot);
                let byte_len = header.slot_size_bytes as usize;
                let frame_slice = unsafe {
                    std::slice::from_raw_parts(self.shm_ptr.as_ptr().add(offset), byte_len)
                };
                Ok(Some((slot, frame_slice)))
            }
            WorkerAck::FrameSkipped { .. } => Ok(None),
        }
    }

    fn handle_worker_crash(&mut self, err: anyhow::Error) -> Result<Option<(u8, &[u8])>> {
        if self.crash_restarts >= MAX_CRASH_RESTARTS {
            self.suspended = true;
            anyhow::bail!(
                "Rust wallpaper {} crashed {} times ({}); suspending worker",
                self.display_name,
                self.crash_restarts,
                err
            );
        }

        if let Some(last) = self.last_restart_instant {
            if last.elapsed() < CRASH_RESTART_COOLDOWN {
                return Ok(None);
            }
        }

        self.crash_restarts += 1;
        self.last_restart_instant = Some(Instant::now());
        warn!(
            "Rust wallpaper {} pipe lost ({}); attempting restart {}/{}",
            self.display_name, err, self.crash_restarts, MAX_CRASH_RESTARTS
        );
        self.respawn_with_dimensions(self.width, self.height)?;
        Ok(None)
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
            let _ = self.child_stdin.write_all(b"\n");
            let _ = self.child_stdin.flush();
        }
    }
}

impl Drop for RustWallpaperRunner {
    fn drop(&mut self) {
        let _ = serde_json::to_writer(&mut self.child_stdin, &WorkerEvent::Terminate);
        let _ = self.child_stdin.write_all(b"\n");
        let _ = self.child_stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
        unsafe {
            let _ = munmap(self.shm_ptr.as_ptr().cast(), self.shm_size);
        }
        warn!(
            "Rust wallpaper process {} stopped, SHM resources released",
            self.display_name
        );
    }
}

//! Supervisor for compiled Rust wallpaper binaries (`RustWallpaperRunner`).
//!
//! `v0.2.0-beta.1` changes:
//! 1. Uses the safe wrappers [`rustix::fs::memfd_create`], [`rustix::fs::ftruncate`] and [`rustix::mm::mmap`].
//! 2. Keeps the **embedded ELF image (`embedded_elf_fd: Option<OwnedFd>`)** alive inside the daemon.
//!    When the user runs `rwallpprs run-embedded`, the daemon creates a long-lived `memfd` holding the
//!    wallpaper ELF and executes `/proc/self/fd/<fd>`, so the CLI can exit immediately while the daemon
//!    stays able to restart the embedded binary at any time.

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
    /// Holds the ELF image of a wallpaper started via `run-embedded` in daemon RAM.
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
    /// Runs an external compiled Rust wallpaper binary from a path on disk.
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
        Self::spawn_internal(display_name, bin_path, None, args, mode, target_fps, width, height)
    }

    /// Runs a wallpaper ELF binary embedded in the CLI straight from daemon memory (`memfd_create`).
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

        // Frame buffer in shared memory without CLOEXEC so the child process inherits the FD
        let shm_fd = memfd_create("rwallpprs-shm-framebuf", MemfdFlags::empty())
            .context("rustix::fs::memfd_create failed for the frame buffer")?;

        ftruncate(&shm_fd, shm_size as u64)
            .context("rustix::fs::ftruncate failed for the SHM buffer")?;

        // Clear FD_CLOEXEC explicitly on the SHM descriptor we hand over
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

        let mut cmd = Command::new(&exec_path);
        cmd.args(&args)
            .env("RWALLPPRS_SHM_FD", shm_fd.as_raw_fd().to_string())
            .env("RWALLPPRS_SHM_SIZE", shm_size.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to launch Rust wallpaper {}", display_name))?;

        let child_stdin = child
            .stdin
            .take()
            .context("Failed to take the child process stdin")?;
        let child_stdout = BufReader::new(
            child
                .stdout
                .take()
                .context("Failed to take the child process stdout")?,
        );

        info!(
            "Started Rust wallpaper {} (PID={}, mode={:?}, SHM={} KB)",
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

    /// Raw FD of the engine frame-buffer `memfd` (zero-copy direct present over the
    /// same file: the daemon imports it into `wl_shm` without copying pixels).
    /// The runner always owns this memfd, so the `RawFd` is not optional.
    pub fn shm_raw_fd(&self) -> RawFd {
        self.shm_fd.as_raw_fd()
    }

    /// Slot that will be used for the **next** frame, without advancing
    /// `current_slot`. Used by main-loop flow control: before sending
    /// `RenderFrame` it checks whether the compositor released that slot.
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
            anyhow::bail!("The Rust wallpaper process closed its reply pipe");
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
                // Return the slot together with the slice: direct present binds the
                // wl_buffer to this exact offset (zero copy through SHM)
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
            "Rust wallpaper process {} stopped, SHM resources released",
            self.display_name
        );
    }
}

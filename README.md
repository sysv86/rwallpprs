# RWallpprs `v0.2.0-beta.2` — Video and Native Rust Wallpapers for Niri (Arch Linux)

**RWallpprs** is a live and procedural wallpaper engine for the scrollable-tiling Wayland compositor
**Niri**, built for **Arch Linux**.

It renders onto a `wlr-layer-shell` background surface and reacts to Niri's compositor state
(horizontal strip scroll, active workspace, Overview, fullscreen windows), so a wallpaper can pan
together with the tiling strip instead of sitting still behind it.

Two kinds of wallpaper are supported:

- **Video** — decoded in-process with `ffmpeg-next` (H.264, HEVC, VP9, AV1) through VA-API, Vulkan,
  NVDEC/CUDA or VDPAU, with an automatic software fallback.
- **Native Rust** — your own wallpaper binary written against the RWallpprs SDK, on the CPU
  (`Canvas`, zero-copy shared memory) or on the GPU (`wgpu` with WGSL shaders, or full 2D scenes
  with your own pipelines and instancing).

## Requirements

- **Niri** on Wayland (`NIRI_SOCKET` must be set in the session).
- **Arch Linux** or an Arch-based distribution (CachyOS, EndeavourOS, Manjaro, Garuda, …).
- A GPU with working **VA-API** or **Vulkan** drivers for hardware decoding and GPU wallpapers.
- `rust` (stable), `clang`, `pkgconf`, `ffmpeg`, `libva`, `vulkan-icd-loader`, `wayland`,
  `wayland-protocols`.

## Install

### Automatic (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/sysv86/rwallpprs/main/install.sh | bash
```

The script checks that the distribution is Arch-based, installs the dependencies with `pacman`,
builds the workspace, installs the binaries, wallpapers, default configuration and the systemd user
unit, and prints the exact Niri startup lines to add.

Options: `--prefix DIR`, `--wallpapers`, `--no-deps`, `--no-systemd`, `--uninstall`.
Run `./install.sh --help` for details.

### From a package

```bash
cd packaging/arch
makepkg -si
```

### Manual

```bash
sudo pacman -S --needed base-devel rust clang pkgconf ffmpeg libva libva-mesa-driver \
    vulkan-icd-loader wayland wayland-protocols

cargo build --release

./target/release/rwallpprs daemon --config config/rwallpprs.toml &
./target/release/rwallpprs set-bin ./target/release/niri-cyber-grid --mode animated --fps 60
```

## Usage

```bash
# Start the daemon (control socket at $XDG_RUNTIME_DIR/rwallpprs.sock)
rwallpprs daemon --config config/rwallpprs.toml &

# Video wallpaper: H.264 / HEVC / VP9 / AV1 through the native ffmpeg-next decoder
rwallpprs set-video ~/Videos/wallpaper_av1.mp4 --hw-accel vaapi --fps 60

# Run a compiled Rust wallpaper
rwallpprs set-bin ./target/release/niri-cyber-grid --mode animated --fps 60

# Embed a wallpaper binary inside the CLI, then run it from daemon memory
rwallpprs bundle-bin ./target/release/niri-voronoi-flow -o ./rwallpprs-voronoi
./rwallpprs-voronoi run-embedded --mode animated --fps 60

# Inspect and control
rwallpprs status
rwallpprs power --toggle-pause
rwallpprs stop
```

`bundle-bin` appends the wallpaper ELF to a copy of the CLI. At `run-embedded` the CLI streams that
payload over the control socket and the daemon keeps it in a long-lived `memfd`, so the binary can be
restarted later (for example after a monitor resolution change) without the CLI staying alive.

### Modes

Wallpapers run in one of three modes (`--mode`):

| Mode | Behavior |
| --- | --- |
| `animated` | Continuous rendering at `--fps`. |
| `static` | Exactly one frame, at startup or on resolution change. Idle cost is 0% CPU/GPU. |
| `reactive` | Redraw only when Niri events arrive: strip scroll, workspace switch, Overview. |

### Power saving

`rwallpprs power` toggles individual rules:

```bash
rwallpprs power --enabled on --auto-hide on
rwallpprs power --pause-fullscreen on --pause-maximized on --wake-overview on
rwallpprs power --battery-limit 25
```

- `--pause-fullscreen` — hide the wallpaper while a fullscreen window is present.
- `--pause-maximized` — hide the wallpaper once the active Niri column covers
  `maximized_column_threshold` (default `0.96`) of the monitor width. Coverage is the sum of the
  strip's column widths, so two or three windows that together fill the screen also trigger it;
  a stack of windows shares one column.
- `--wake-overview` — always wake the wallpaper in Niri Overview (`Mod+O`).
- `--battery-limit` — freeze on the last frame while on battery below the given percentage.

The surface is deliberately **not** detached when hidden: detaching the buffer from a
`zwlr-layer-shell` surface that uses `L/R + T/B` anchors makes Niri raise a protocol error and
destroys the layer permanently. Hidden means "engine paused, last frame kept".

## Configuration

`config/rwallpprs.toml` is installed to `/etc/xdg/rwallpprs/rwallpprs.toml`. Every field has a
default, so the daemon runs with no configuration file at all.

```toml
[general]
layer = "background"
log_level = "info"

[video]
hw_accel = "auto"                       # auto | vaapi | vulkan | nvdec | vdpau | software
drm_render_node = "/dev/dri/renderD128"
scale_mode = "fill"                     # fill | fit | stretch
fps_limit = 60
loop_playback = true

[rust_engine]
default_mode = "animated"               # animated | static | reactive_niri
target_fps = 60
double_buffering = true
forward_niri_events = true

[power_saving]
enabled = true
auto_hide_surface = true
pause_on_fullscreen = true
pause_on_maximized_column = true
maximized_column_threshold = 0.96
wake_on_niri_overview = true
battery_freeze_threshold = 25
```

Environment variables:

- `RUST_LOG` — tracing filter, for example `RUST_LOG=debug` or `RUST_LOG=rwallpprs_daemon=trace`.
- `NIRI_SOCKET` — required for compositor events; without it the wallpaper still renders, but does
  not react to scrolling or Overview.
- `RWALLPPRS_GPU_BACKEND` — GPU backend selection for `wgpu` wallpapers (`vulkan`, `gl`, or `auto`).
- `RWALLPPRS_GPU_POWER` — adapter power preference (`low` default, or `high` for discrete GPUs).
- `RWALLPPRS_GPU_SYNC_READBACK` — set to `1` to force single-buffer synchronous GPU readback.
- `RWALLPPRS_GPU_VALIDATION` — set to `1` to enable `wgpu` validation layers for shader debugging.

## Niri integration

Add the startup lines and bindings from `packaging/niri/config.kdl.snippet` to
`~/.config/niri/config.kdl`:

```kdl
spawn-at-startup "rwallpprs" "daemon"
spawn-at-startup "rwallpprs" "set-bin" "/usr/lib/rwallpprs/wallpapers/niri-cyber-grid" "--mode" "animated" "--fps" "60"

layout {
    background-color "transparent"
}

binds {
    Mod+Shift+W { spawn "rwallpprs" "power" "--toggle-pause"; }
    Mod+Alt+1   { spawn "rwallpprs" "set-bin" "/usr/lib/rwallpprs/wallpapers/niri-cyber-grid"; }
    Mod+Alt+2   { spawn "rwallpprs" "set-bin" "/usr/lib/rwallpprs/wallpapers/niri-voronoi-flow"; }
    Mod+Alt+3   { spawn "rwallpprs" "set-bin" "/usr/lib/rwallpprs/wallpapers/minimal-arch-static" "--mode" "static"; }
}
```

`background-color "transparent"` is required — otherwise Niri paints over the layer surface.

The systemd user unit is installed to `$PREFIX/lib/systemd/user/rwallpprs.service`:

```bash
systemctl --user daemon-reload
systemctl --user enable --now rwallpprs.service
```

## Writing wallpapers

The SDK crate `rwallpprs-sdk` supports three styles:

```rust
// CPU: draw straight into Wayland shared memory through a Canvas.
rwallpprs_sdk::run_wallpaper!(MyCanvasWallpaper);

// GPU: one WGSL fragment shader, with uniforms for Niri state.
rwallpprs_sdk::run_shader_wallpaper!(MyShaderWallpaper);

// GPU: a full wgpu scene (own pipelines, buffers, instancing, extra passes).
rwallpprs_sdk::run_scene_wallpaper!(MySceneWallpaper);
```

All three receive `FrameContext` with `time_secs`, `dt_secs`, the frame size and a
`NiriViewportState` (strip scroll offset, active workspace, Overview state, screen coverage).
An offline environment renders one frame and exits, which makes wallpapers testable without a
compositor.

Bundled examples:

- `examples/minimal-arch-static` — the smallest CPU wallpaper: a gradient and a static frame.
- `examples/niri-cyber-grid` — WGSL shader wallpaper panning with the Niri strip.
- `examples/niri-voronoi-flow` — procedural CPU wallpaper.

See the crate documentation in `crates/rwallpprs-sdk/src/lib.rs` and
`crates/rwallpprs-sdk/src/wgpu_renderer.rs` for the full API.

## Companion wallpaper: Android 17 "Landroid"

The Android 17 Easter Egg screensaver, ported to a GPU `wgpu` scene by this SDK, lives in its own
repository. To install and run it in one command:

```bash
curl -fsSL https://raw.githubusercontent.com/sysv86/niri-landroid/main/install.sh | bash
```

That script installs the build dependencies, builds the wallpaper against `rwallpprs-sdk`, installs
it, and starts it on the Niri desktop.

## How it works

Crates in this workspace:

| Crate | Role |
| --- | --- |
| `rwallpprs-ipc` | Wire protocol: control socket (u32 length prefix + bincode), worker events, `memfd` frame header. |
| `rwallpprs-sdk` | Wallpaper SDK: `Canvas`, `GpuScene`, the `Wallpaper` / `GpuShaderWallpaper` / `GpuSceneWallpaper` traits and their launch macros. |
| `rwallpprs-daemon` | Wayland layer surface, Niri event monitor, video decoder, wallpaper supervisor, power manager. |
| `rwallpprs-cli` | User-facing `rwallpprs` command, including `bundle-bin` / `run-embedded`. |

Design points worth knowing:

- **Zero-copy presentation.** The `wl_shm` pool is created over the same `memfd` the engine renders
  into, so a frame reaches the compositor without a single `memcpy`. Two slots alternate on
  `wl_buffer.release`; the main loop falls back to a copying path when direct presentation is
  unavailable, so output degrades without stalling.
- **Flow control before decode.** A frame is only decoded into a slot the compositor has already
  released, so no ffmpeg frame is wasted and no buffer still in use is overwritten.
- **Native multimedia stack.** `ffmpeg-next` / `ffmpeg-sys-next` are called in-process; there is no
  `Command::new("ffmpeg")` anywhere. Hardware surfaces are downloaded with
  `av_hwframe_transfer_data` and scaled by `libswscale` straight into the `XRGB8888` frame buffer.
- **Niri IPC through the official crate.** `niri-ipc` provides typed events; the viewport ratio is
  computed against the real monitor size reported by `zwlr_layer_surface_v1::configure`, not a
  hardcoded resolution.
- **Safe syscalls.** `memfd_create`, `ftruncate`, `mmap` and `fstat` go through `rustix` with
  `OwnedFd` handles instead of raw `libc` calls.
- **Persistent embedded binaries.** `run-embedded` hands the ELF to the daemon over the socket; the
  daemon owns a long-lived `memfd` and executes `/proc/self/fd/<fd>`, so the CLI can exit immediately
  and the daemon can still restart the wallpaper later.

## Troubleshooting

**The wallpaper never appears.** Check that `layout { background-color "transparent" }` is set in
`config.kdl`, then run `rwallpprs status`: the `wl_surface visible` flag on the monitor line tells you
whether frames are being presented.

**No Niri events (`NIRI_SOCKET active: false`).** The daemon was started outside the session that
owns `NIRI_SOCKET`. Start it from Niri (`spawn-at-startup`) or export `NIRI_SOCKET` first.

**Hardware decoding fails.** The daemon logs which backend it selected and falls back to software
automatically. Install the driver matching your GPU (`libva-mesa-driver` and `vulkan-radeon` for AMD,
`intel-media-driver` and `vulkan-intel` for Intel) and check that
`/dev/dri/renderD128` exists.

**CPU stays high while idle.** Use `--mode static` for wallpapers that do not animate, or
`--mode reactive` for wallpapers that only need to follow the Niri strip.

**Collecting logs.**

```bash
RUST_LOG=debug rwallpprs daemon --config config/rwallpprs.toml
```

## License

MIT. See `LICENSE`.

#!/usr/bin/env bash
#
# RWallpprs automatic installer for Arch Linux and Arch-based distributions
# (CachyOS, EndeavourOS, Manjaro, Garuda, Artix with pacman, ...).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/sysv86/rwallpprs/main/install.sh | bash
#   ./install.sh [options]
#
# Options:
#   --prefix DIR     Install prefix for binaries and libraries (default: /usr/local)
#   --no-deps        Do not run pacman; assume every dependency is present
#   --no-systemd     Do not install the systemd user unit
#   --uninstall      Remove an installation made by this script
#   -h, --help       Show this help
#
# The script installs:
#   $PREFIX/bin/rwallpprs                  CLI
#   $PREFIX/bin/rwallpprs-daemon           daemon
#   $PREFIX/lib/rwallpprs/wallpapers/*     bundled Rust wallpapers
#   /etc/xdg/rwallpprs/rwallpprs.toml      default configuration
#   $PREFIX/lib/systemd/user/rwallpprs.service

set -Eeuo pipefail

readonly SELF_URL="https://raw.githubusercontent.com/sysv86/rwallpprs/main/install.sh"
readonly REPO_URL="https://github.com/sysv86/rwallpprs.git"

PREFIX="/usr/local"
INSTALL_DEPS=1
INSTALL_SYSTEMD=1
MODE="install"

WALLPAPERS=(
    niri-cyber-grid
    niri-voronoi-flow
    minimal-arch-static
)

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    # Print the leading comment block, up to the first blank line.
    sed -n '3,/^$/p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' || true
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)     [[ -n "${2:-}" ]] || die "--prefix needs a directory"; PREFIX="$2"; shift 2 ;;
        --prefix=*)   PREFIX="${1#*=}"; shift ;;
        --no-deps)    INSTALL_DEPS=0; shift ;;
        --no-systemd) INSTALL_SYSTEMD=0; shift ;;
        --uninstall)  MODE="uninstall"; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            die "unknown option: $1 (try --help)" ;;
    esac
done

# ---------------------------------------------------------------------------
# When the script is piped into bash, stdin is the script itself. Any command
# that reads from stdin (sudo's password prompt) would swallow the remaining
# script text, so re-exec from a real file first.
# ---------------------------------------------------------------------------
if [[ ! -t 0 && -z "${RWALLPPRS_INSTALL_REEXEC:-}" ]]; then
    export RWALLPPRS_INSTALL_REEXEC=1
    tmp="$(mktemp /tmp/rwallpprs-install.XXXXXX.sh)"
    curl -fsSL "$SELF_URL" -o "$tmp" || die "could not download $SELF_URL"
    if { : </dev/tty; } 2>/dev/null; then
        exec bash "$tmp" "$@" </dev/tty
    fi
    exec bash "$tmp" "$@"
fi

[[ "$(uname -s)" == "Linux" ]] || die "RWallpprs targets Linux (Niri runs on Linux only)"

if ! command -v pacman >/dev/null 2>&1; then
    die "this script supports Arch-based distributions only (pacman not found).
On other distributions install the dependencies listed in README.md manually,
then run: cargo build --release && sudo make install"
fi

if [[ "${EUID}" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "run as root or install sudo"
    SUDO="sudo"
else
    SUDO=""
fi

asroot() { $SUDO "$@"; }

# ---------------------------------------------------------------------------
# Locate sources: either a local checkout or a fresh clone.
# ---------------------------------------------------------------------------
if [[ -f "${PWD}/Cargo.toml" && -d "${PWD}/crates/rwallpprs-daemon" ]]; then
    SRC_DIR="$PWD"
    CLEANUP=""
else
    command -v git >/dev/null 2>&1 || die "git is required to fetch the sources"
    SRC_DIR="$(mktemp -d /tmp/rwallpprs-src.XXXXXX)"
    CLEANUP="$SRC_DIR"
fi

cleanup() { if [[ -n "${CLEANUP:-}" ]]; then rm -rf "$CLEANUP"; fi; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
if [[ "$MODE" == "uninstall" ]]; then
    log "Stopping the daemon if it is running"
    asroot systemctl --user stop rwallpprs.service 2>/dev/null || true
    log "Removing installed files under $PREFIX"
    asroot rm -f "$PREFIX/bin/rwallpprs" "$PREFIX/bin/rwallpprs-daemon"
    asroot rm -f "$PREFIX/lib/systemd/user/rwallpprs.service"
    asroot rm -rf "$PREFIX/lib/rwallpprs"
    log "Configuration kept at /etc/xdg/rwallpprs/rwallpprs.toml (remove it manually if you want)"
    log "Done"
    exit 0
fi

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
readonly DEPS=(
    base-devel
    rust
    clang
    pkgconf
    git
    ffmpeg
    libva
    vulkan-icd-loader
    wayland
    wayland-protocols
)

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    log "Installing build and runtime dependencies"
    asroot pacman -S --needed --noconfirm "${DEPS[@]}"
else
    log "Skipping dependency installation (--no-deps)"
fi

# Optional GPU vendor drivers: nice to have, never fatal.
case "$(lspci 2>/dev/null | grep -i -m1 -E 'vga|3d|display' || true)" in
    *AMD*|*Advanced\ Micro\ Devices*)
        warn "AMD GPU detected: install libva-mesa-driver and vulkan-radeon for hardware decoding"
        ;;
    *Intel*)
        warn "Intel GPU detected: install intel-media-driver and vulkan-intel for hardware decoding"
        ;;
esac

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [[ -z "$CLEANUP" ]]; then
    log "Building from the current checkout at $SRC_DIR"
else
    log "Cloning $REPO_URL"
    git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

cd "$SRC_DIR"
log "Building the workspace in release mode (this can take several minutes)"
cargo build --release --locked

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
log "Installing binaries into $PREFIX/bin"
asroot install -Dm755 target/release/rwallpprs        "$PREFIX/bin/rwallpprs"
asroot install -Dm755 target/release/rwallpprs-daemon "$PREFIX/bin/rwallpprs-daemon"

log "Installing example wallpapers into $PREFIX/lib/rwallpprs/wallpapers"
for wp in "${WALLPAPERS[@]}"; do
    asroot install -Dm755 "target/release/${wp}" "$PREFIX/lib/rwallpprs/wallpapers/${wp}"
done

if [[ ! -f /etc/xdg/rwallpprs/rwallpprs.toml ]]; then
    log "Installing the default configuration to /etc/xdg/rwallpprs/rwallpprs.toml"
    asroot install -Dm644 config/rwallpprs.toml /etc/xdg/rwallpprs/rwallpprs.toml
else
    log "Keeping the existing configuration at /etc/xdg/rwallpprs/rwallpprs.toml"
fi

if [[ "$INSTALL_SYSTEMD" -eq 1 ]]; then
    log "Installing the systemd user unit"
    asroot install -Dm644 packaging/systemd/rwallpprs.service \
        "$PREFIX/lib/systemd/user/rwallpprs.service"
    # The unit ships with /usr/bin; rewrite ExecStart for a custom prefix.
    if [[ "$PREFIX" != "/usr" ]]; then
        asroot sed -i "s#^ExecStart=/usr/bin/rwallpprs-daemon#ExecStart=$PREFIX/bin/rwallpprs-daemon#" \
            "$PREFIX/lib/systemd/user/rwallpprs.service"
    fi
fi

cat <<EOF

RWallpprs is installed.

Start it inside a running Niri session:

    rwallpprs daemon --config /etc/xdg/rwallpprs/rwallpprs.toml &
    rwallpprs set-bin $PREFIX/lib/rwallpprs/wallpapers/niri-cyber-grid --mode animated --fps 60

Or enable the systemd user service:

    systemctl --user daemon-reload
    systemctl --user enable --now rwallpprs.service

Add the startup lines and key bindings from packaging/niri/config.kdl.snippet
to ~/.config/niri/config.kdl.

Want to try the Android-17 "Landroid" wallpaper? One command:

    curl -fsSL https://raw.githubusercontent.com/sysv86/niri-landroid/main/install.sh | bash
EOF

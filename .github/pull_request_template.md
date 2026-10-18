## What changed

<!-- One paragraph. What is different after this PR, from the user's point of view. -->

## Why

<!-- The problem this solves, or the behaviour that was wrong. Link the issue: Fixes #123 -->

## How

<!-- The approach, and why it is the right one. Note anything a reviewer might disagree with. -->

## Affected areas

<!-- Tick everything the diff touches. -->

- [ ] `rwallpprs-ipc` (wire protocol — breaking change?)
- [ ] `rwallpprs-sdk` (public wallpaper API)
- [ ] `rwallpprs-daemon` (Wayland, Niri events, video engine, power management)
- [ ] `rwallpprs-cli`
- [ ] Examples
- [ ] Packaging, install script, systemd unit
- [ ] Documentation

## Testing

<!-- Be exact. State what you ran and what it printed. Do not claim tests that were not run. -->

```
cargo build --release
cargo clippy --all-targets -- -D warnings
```

Hardware and session used for manual verification (GPU, driver, Niri version):

## Checklist

- [ ] The change is scoped to the problem; unrelated files are untouched.
- [ ] `cargo build --release` succeeds.
- [ ] Behaviour changes are reflected in `README.md` or the relevant doc comments.
- [ ] New configuration keys or CLI flags have defaults, so existing setups keep working.
- [ ] No secrets, build artifacts or editor files are included.

## Screenshots or recordings

<!-- Required for anything visual: a wallpaper, a rendering change, an effect on the compositor. -->

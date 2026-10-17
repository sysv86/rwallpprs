use rwallpprs_sdk::{run_wallpaper, Canvas, Color, FrameContext, RustWallpaperMode, Wallpaper};

pub struct VoronoiFlowWallpaper {
    seeds: [(f32, f32, f32, f32); 12],
}

impl Wallpaper for VoronoiFlowWallpaper {
    fn init(_args: &[String]) -> Self {
        let mut seeds = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 12];
        for (i, s) in seeds.iter_mut().enumerate() {
            let fi = i as f32 + 1.0;
            *s = (
                (fi * 1.618).fract(),
                (fi * 2.718).fract(),
                ((fi * 0.73).sin()) * 0.12,
                ((fi * 1.37).cos()) * 0.12,
            );
        }
        Self { seeds }
    }

    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    fn render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> bool {
        let t = ctx.time_secs as f32;
        let ws = ctx.niri.active_workspace_idx;

        let (base_col, edge_col) = match ws % 4 {
            0 => (Color::from_hex(0x0B132B), Color::from_hex(0x5BC0BE)),
            1 => (Color::from_hex(0x1A1423), Color::from_hex(0xE07A5F)),
            2 => (Color::from_hex(0x081C15), Color::from_hex(0x52B788)),
            _ => (Color::from_hex(0x10002B), Color::from_hex(0xC77DFF)),
        };

        let mut animated_seeds = [(0.0f32, 0.0f32); 12];
        for (i, &(bx, by, vx, vy)) in self.seeds.iter().enumerate() {
            let px = (bx + (t * vx + i as f32).sin() * 0.25).rem_euclid(1.0);
            let py = (by + (t * vy + i as f32 * 0.5).cos() * 0.25).rem_euclid(1.0);
            animated_seeds[i] = (px, py);
        }

        canvas.shade_uv(|u, v| {
            let mut d1 = 10.0f32;
            let mut d2 = 10.0f32;
            for &(sx, sy) in &animated_seeds {
                let dx = u - sx;
                let dy = v - sy;
                let d = (dx * dx + dy * dy).sqrt();
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                } else if d < d2 {
                    d2 = d;
                }
            }
            let border = (1.0 - ((d2 - d1) * 18.0)).clamp(0.0, 1.0);
            base_col.lerp(edge_col, border * border)
        });

        true
    }
}

run_wallpaper!(VoronoiFlowWallpaper);

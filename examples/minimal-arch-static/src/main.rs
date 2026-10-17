use rwallpprs_sdk::{run_wallpaper, Canvas, Color, FrameContext, RustWallpaperMode, Wallpaper};

pub struct MinimalArchStatic {
    last_size: (u32, u32),
}

impl Wallpaper for MinimalArchStatic {
    fn init(_args: &[String]) -> Self {
        Self { last_size: (0, 0) }
    }

    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Static
    }

    fn render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) -> bool {
        if self.last_size == (ctx.width, ctx.height) && ctx.frame_index > 0 {
            return false;
        }
        self.last_size = (ctx.width, ctx.height);

        let top = Color::from_hex(0x0D1117);
        let bottom = Color::from_hex(0x161B22);
        let arch_blue = Color::from_hex(0x1793D1);
        let arch_cyan = Color::from_hex(0x38BDF8);

        canvas.fill_vertical_gradient(top, bottom);

        let cx = ctx.width as f32 * 0.5;
        let cy = ctx.height as f32 * 0.5;
        let r = (ctx.height as f32) * 0.22;

        canvas.fill_circle_sdf(cx, cy, r, arch_blue.lerp(bottom, 0.72));
        canvas.fill_circle_sdf(cx, cy, r * 0.82, bottom);

        let apex_y = (cy - r * 0.65) as i32;
        let left_x = (cx - r * 0.52) as i32;
        let right_x = (cx + r * 0.52) as i32;
        let base_y = (cy + r * 0.55) as i32;

        for offset in -2..=2 {
            canvas.draw_line(cx as i32 + offset, apex_y, left_x + offset, base_y, arch_cyan);
            canvas.draw_line(cx as i32 + offset, apex_y, right_x + offset, base_y, arch_cyan);
            canvas.draw_line(left_x, base_y + offset, right_x, base_y + offset, arch_blue);
        }

        true
    }
}

run_wallpaper!(MinimalArchStatic);

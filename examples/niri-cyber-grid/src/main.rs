//! Hardware GPU wallpaper `niri-cyber-grid` built on `wgpu` (Vulkan/GL WGSL shader)
//! with an automatic CPU fallback.
//!
//! Reads `u.niri_scroll_offset` and `u.niri_overview_open` directly inside the WGSL shader!

use rwallpprs_sdk::{
    run_shader_wallpaper, Canvas, Color, FrameContext, GpuShaderWallpaper, RustWallpaperMode,
};

pub struct CyberGridShaderWallpaper {
    smooth_scroll_x: f32,
}

const CYBER_GRID_WGSL: &str = r#"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let scroll_x = u.niri_scroll_offset * 0.002;
    let horizon = 0.52 - u.niri_overview_open * 0.08;

    let sky_top = vec3<f32>(0.027, 0.020, 0.063);
    let sky_horizon = vec3<f32>(0.165, 0.035, 0.267);
    let ground_base = vec3<f32>(0.035, 0.016, 0.078);
    let neon_cyan = vec3<f32>(0.0, 0.96, 0.83);
    let neon_magenta = vec3<f32>(1.0, 0.165, 0.52);

    if (in.uv.y < horizon) {
        let t = clamp(in.uv.y / horizon, 0.0, 1.0);
        var col = mix(sky_top, sky_horizon, t * t);

        let sun_center = vec2<f32>(0.5 - scroll_x * 0.15, horizon - 0.12);
        let d = length(vec2<f32>((in.uv.x - sun_center.x) * aspect, in.uv.y - sun_center.y));
        let sun_r = 0.16 + u.niri_overview_open * 0.03;
        if (d < sun_r) {
            let scan = sin(in.uv.y * 90.0 - u.time * 2.0);
            if (scan > -0.25 || in.uv.y < sun_center.y) {
                let sg = clamp((in.uv.y - (sun_center.y - sun_r)) / (2.0 * sun_r), 0.0, 1.0);
                col = mix(vec3<f32>(1.0, 0.745, 0.043), vec3<f32>(0.984, 0.337, 0.027), sg);
            }
        }
        return vec4<f32>(col, 1.0);
    } else {
        let dz = max(in.uv.y - horizon, 0.002);
        let depth = 0.25 / dz;
        let wx = (in.uv.x - 0.5) * aspect * depth + scroll_x * 4.0;
        let wz = depth + u.time * (1.8 + u.niri_overview_open * 1.5);

        let gx = abs(fract(wx) - 0.5);
        let gz = abs(fract(wz) - 0.5);
        let line_mask = clamp(max((0.48 - gx) * 35.0, (0.47 - gz) * 28.0), 0.0, 1.0);
        let fog = clamp(exp(-depth * 0.12), 0.0, 1.0);
        let grid_col = mix(neon_cyan, neon_magenta, clamp(dz * 2.2, 0.0, 1.0));
        let final_col = mix(ground_base, grid_col, line_mask * fog);
        return vec4<f32>(final_col, 1.0);
    }
}
"#;

impl GpuShaderWallpaper for CyberGridShaderWallpaper {
    fn init(_args: &[String]) -> Self {
        Self {
            smooth_scroll_x: 0.0,
        }
    }

    fn wgsl_fragment_source(&self) -> &str {
        CYBER_GRID_WGSL
    }

    fn default_mode(&self) -> RustWallpaperMode {
        RustWallpaperMode::Animated
    }

    fn fallback_cpu_render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) {
        let target_scroll = ctx.niri.strip_scroll_offset as f32 * 0.002;
        self.smooth_scroll_x +=
            (target_scroll - self.smooth_scroll_x) * (ctx.dt_secs * 8.0).clamp(0.0, 1.0);
        let scroll_x = self.smooth_scroll_x;
        let t = ctx.time_secs as f32;
        let horizon = if ctx.niri.overview_open { 0.44 } else { 0.52 };
        let aspect = (ctx.width as f32) / (ctx.height.max(1) as f32);

        let sky_top = Color::from_hex(0x070510);
        let sky_horizon = Color::from_hex(0x2A0944);
        let ground_base = Color::from_hex(0x090414);
        let neon_cyan = Color::from_hex(0x00F5D4);
        let neon_magenta = Color::from_hex(0xFF2A85);

        canvas.shade_uv(|u, v| {
            if v < horizon {
                let sky_t = (v / horizon).clamp(0.0, 1.0);
                sky_top.lerp(sky_horizon, sky_t * sky_t)
            } else {
                let dz = (v - horizon).max(0.002);
                let depth = 0.25 / dz;
                let wx = (u - 0.5) * aspect * depth + scroll_x * 4.0;
                let wz = depth + t * 1.8;
                let gx = (wx.fract().abs() - 0.5).abs();
                let gz = (wz.fract().abs() - 0.5).abs();
                let mask = ((0.48 - gx).max(0.0) * 35.0)
                    .max((0.47 - gz).max(0.0) * 28.0)
                    .clamp(0.0, 1.0);
                let fog = (-depth * 0.12).exp().clamp(0.0, 1.0);
                let grid_color = neon_cyan.lerp(neon_magenta, (dz * 2.2).clamp(0.0, 1.0));
                ground_base.lerp(grid_color, mask * fog)
            }
        });
    }
}

run_shader_wallpaper!(CyberGridShaderWallpaper);

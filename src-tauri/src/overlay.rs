//! The recording overlay: a small pill near the bottom of the screen that shows
//! whispr is listening, then transcribing, then what happened. Hotkey sessions
//! run with the main window hidden, so without it there is no sign that the
//! microphone is open.
//!
//! It must never take focus - the transcript is pasted into whatever window
//! has focus, so the overlay stealing it would paste into the overlay. It is
//! also click-through, so it never blocks the app underneath.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tauri::{AppHandle, Manager, PhysicalPosition, WebviewUrl, WebviewWindowBuilder};

pub const LABEL: &str = "overlay";

/// Logical size of the pill, matched by overlay.html.
const WIDTH: f64 = 200.0;
const HEIGHT: f64 = 48.0;
/// Gap between the pill and the bottom of the work area (above the taskbar).
const BOTTOM_GAP: f64 = 64.0;

/// Bumped on every show, so a delayed hide from an earlier session cannot hide
/// the overlay of a session that started since.
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn create(app: &AppHandle) -> tauri::Result<()> {
    let builder = WebviewWindowBuilder::new(app, LABEL, WebviewUrl::App("overlay.html".into()))
        .title("whispr")
        .inner_size(WIDTH, HEIGHT)
        .resizable(false)
        .decorations(false)
        .shadow(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .focused(false)
        .focusable(false)
        .visible(false);
    // Transparency needs a private API on macOS; see the macOS pass.
    #[cfg(not(target_os = "macos"))]
    let builder = builder.transparent(true);

    let window = builder.build()?;
    window.set_ignore_cursor_events(true)?;
    Ok(())
}

/// Show the overlay on the monitor the pointer is on, which is where the user
/// is working.
pub fn show(app: &AppHandle) {
    GENERATION.fetch_add(1, Ordering::SeqCst);
    let Some(window) = app.get_webview_window(LABEL) else {
        return;
    };
    let monitor = app
        .cursor_position()
        .ok()
        .and_then(|p| app.monitor_from_point(p.x, p.y).ok().flatten())
        .or_else(|| app.primary_monitor().ok().flatten());
    if let Some(monitor) = monitor {
        let area = monitor.work_area();
        let position = place(
            (area.position.x, area.position.y),
            (area.size.width, area.size.height),
            monitor.scale_factor(),
        );
        let _ = window.set_position(position);
    }
    let _ = window.show();
}

/// Hide after `delay`, unless another session has shown the overlay since.
pub fn hide_after(app: &AppHandle, delay: Duration) {
    let generation = GENERATION.load(Ordering::SeqCst);
    let app = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        if GENERATION.load(Ordering::SeqCst) == generation {
            if let Some(window) = app.get_webview_window(LABEL) {
                let _ = window.hide();
            }
        }
    });
}

/// Bottom centre of a work area, in physical pixels.
fn place(origin: (i32, i32), size: (u32, u32), scale: f64) -> PhysicalPosition<i32> {
    let width = (WIDTH * scale).round() as i32;
    let height = (HEIGHT * scale).round() as i32;
    let gap = (BOTTOM_GAP * scale).round() as i32;
    PhysicalPosition::new(
        origin.0 + (size.0 as i32 - width) / 2,
        origin.1 + size.1 as i32 - height - gap,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centred_above_the_bottom_at_100_percent() {
        // 1920x1040 work area: a 1080p screen minus a 40 px taskbar.
        let p = place((0, 0), (1920, 1040), 1.0);
        assert_eq!(p.x, (1920 - 200) / 2);
        assert_eq!(p.y, 1040 - 48 - 64);
    }

    #[test]
    fn scales_with_the_monitor() {
        let p = place((0, 0), (2560, 1400), 1.5);
        assert_eq!(p.x, (2560 - 300) / 2);
        assert_eq!(p.y, 1400 - 72 - 96);
    }

    #[test]
    fn follows_a_secondary_monitor_origin() {
        // A second screen to the left of the primary one has a negative x.
        let p = place((-1920, 0), (1920, 1040), 1.0);
        assert_eq!(p.x, -1920 + (1920 - 200) / 2);
    }
}

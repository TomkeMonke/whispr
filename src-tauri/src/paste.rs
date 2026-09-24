//! Getting the transcript into the focused window: put it on the clipboard, send
//! the paste shortcut, then put back whatever was on the clipboard before.
//!
//! Pasting beats typing the text out key by key: it is instant for long
//! dictations and immune to keyboard layouts and autocorrect in the target app.
//! If the keystroke cannot be sent (macOS without Accessibility access), the
//! text is simply left on the clipboard.

use std::thread;
use std::time::Duration;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};

/// How long the target app gets to read the clipboard before the old contents
/// go back. Paste handling is asynchronous in most apps, and Electron ones are
/// the slowest; restoring too early would paste the old clipboard instead.
const RESTORE_DELAY: Duration = Duration::from_millis(400);

/// The hotkey's modifiers may still be held when the transcript is ready. Ctrl
/// and Shift still down would turn Ctrl+V into Ctrl+Shift+V, which many apps
/// read as "paste as plain text" or something else entirely.
const MODIFIER_WAIT: Duration = Duration::from_millis(1500);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Delivery {
    /// Pasted into the focused window.
    Pasted,
    /// Left on the clipboard: auto-paste is off, or the keystroke failed.
    Copied,
}

/// Blocking: waits on modifier keys and the restore delay. Run it off the
/// async runtime.
pub fn deliver(text: &str, auto_paste: bool) -> Result<Delivery, String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| format!("clipboard: {e}"))?;
    // Only text survives the round trip. An image on the clipboard is lost,
    // which is an acceptable price for not clobbering the far commoner text case.
    let previous = clipboard.get_text().ok();
    clipboard
        .set_text(text)
        .map_err(|e| format!("clipboard: {e}"))?;

    if !auto_paste {
        return Ok(Delivery::Copied);
    }

    wait_for_modifiers_released(MODIFIER_WAIT);
    if let Err(e) = send_paste() {
        eprintln!("[whispr] paste keystroke failed, leaving text on the clipboard: {e}");
        return Ok(Delivery::Copied);
    }

    thread::sleep(RESTORE_DELAY);
    if let Some(previous) = previous {
        let _ = clipboard.set_text(previous);
    }
    Ok(Delivery::Pasted)
}

fn send_paste() -> Result<(), String> {
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;

    #[cfg(target_os = "macos")]
    let (modifier, v) = (Key::Meta, Key::Unicode('v'));
    // Key::V is the virtual key, which works whatever the keyboard layout.
    #[cfg(target_os = "windows")]
    let (modifier, v) = (Key::Control, Key::V);
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let (modifier, v) = (Key::Control, Key::Unicode('v'));

    enigo
        .key(modifier, Direction::Press)
        .map_err(|e| e.to_string())?;
    let clicked = enigo.key(v, Direction::Click);
    // Always let go of the modifier, even if the V failed.
    let released = enigo.key(modifier, Direction::Release);
    clicked.and(released).map_err(|e| e.to_string())
}

#[cfg(target_os = "windows")]
fn wait_for_modifiers_released(timeout: Duration) {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    };
    let started = std::time::Instant::now();
    while started.elapsed() < timeout {
        // High bit set means the key is down right now.
        let any_down = [VK_CONTROL, VK_SHIFT, VK_MENU, VK_LWIN, VK_RWIN]
            .iter()
            .any(|&vk| unsafe { GetAsyncKeyState(vk as i32) } < 0);
        if !any_down {
            return;
        }
        thread::sleep(Duration::from_millis(15));
    }
}

/// Unverified off Windows; the transcription itself takes long enough that the
/// hotkey is normally released by the time this runs.
#[cfg(not(target_os = "windows"))]
fn wait_for_modifiers_released(_timeout: Duration) {}

//! Global hotkey: tap to toggle, hold to talk.
//!
//! One key does both. A tap starts recording and leaves it running until the
//! next press; holding past `HOLD_THRESHOLD` records only while the key is down.
//! Either way the transcript goes to whatever window has focus, so the whispr
//! window never has to be shown.
//!
//! On Windows the plugin registers with MOD_NOREPEAT, so holding the key does
//! not fire repeated presses, and it reports the release by polling the key.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager};
use tauri_plugin_global_shortcut::{Shortcut, ShortcutEvent, ShortcutState};

use crate::AppState;

pub const DEFAULT: &str = "Ctrl+Shift+Space";

/// Held longer than this, the key is push-to-talk: releasing it ends the
/// recording. Shorter is a tap, which toggles. 350 ms is comfortably longer
/// than a deliberate tap and shorter than the first word of a sentence.
const HOLD_THRESHOLD: Duration = Duration::from_millis(350);

pub fn parse(combo: &str) -> Option<Shortcut> {
    combo.trim().parse().ok()
}

/// When the press that started the current recording happened.
#[derive(Default)]
pub struct Hotkey {
    pressed_at: Mutex<Option<Instant>>,
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Start,
    Finish,
    Nothing,
}

/// What a key event should do, given where the recorder is. `held_for` is how
/// long ago the press that started the recording happened, if this key started it.
fn decide(pressed: bool, recording: bool, busy: bool, held_for: Option<Duration>) -> Action {
    match (pressed, recording) {
        // A transcription is still running; a new recording would race it.
        (true, _) if busy => Action::Nothing,
        (true, true) => Action::Finish,
        (true, false) => Action::Start,
        (false, true) if held_for.is_some_and(|d| d >= HOLD_THRESHOLD) => Action::Finish,
        (false, _) => Action::Nothing,
    }
}

pub fn handle(app: &AppHandle, event: ShortcutEvent) {
    let state = app.state::<AppState>();
    let pressed = event.state() == ShortcutState::Pressed;
    let mut pressed_at = state.hotkey.pressed_at.lock().unwrap();

    let action = decide(
        pressed,
        state.recorder.is_recording(),
        state.is_busy(),
        pressed_at.map(|t| t.elapsed()),
    );
    // A release always ends this key's claim on the recording's start time:
    // after a tap, the recording carries on until the next press.
    if !pressed {
        *pressed_at = None;
    }

    match action {
        Action::Start => {
            *pressed_at = Some(Instant::now());
            drop(pressed_at);
            crate::start_dictation(app);
        }
        Action::Finish => {
            *pressed_at = None;
            drop(pressed_at);
            let app = app.clone();
            tauri::async_runtime::spawn(async move { crate::finish_dictation(app).await });
        }
        Action::Nothing => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAP: Option<Duration> = Some(Duration::from_millis(120));
    const HOLD: Option<Duration> = Some(Duration::from_millis(900));

    #[test]
    fn tap_toggles() {
        assert_eq!(decide(true, false, false, None), Action::Start);
        // Released quickly: keep recording.
        assert_eq!(decide(false, true, false, TAP), Action::Nothing);
        // Next press ends it; its own release does nothing more.
        assert_eq!(decide(true, true, false, None), Action::Finish);
        assert_eq!(decide(false, false, false, None), Action::Nothing);
    }

    #[test]
    fn hold_is_push_to_talk() {
        assert_eq!(decide(true, false, false, None), Action::Start);
        assert_eq!(decide(false, true, false, HOLD), Action::Finish);
    }

    #[test]
    fn presses_during_transcription_are_ignored() {
        assert_eq!(decide(true, false, true, None), Action::Nothing);
    }

    #[test]
    fn release_of_a_key_that_did_not_start_it_is_ignored() {
        // Recording started from the window button, then the hotkey was released.
        assert_eq!(decide(false, true, false, None), Action::Nothing);
    }

    #[test]
    fn default_combo_parses_and_junk_does_not() {
        assert!(parse(DEFAULT).is_some());
        assert!(parse("ctrl+alt+d").is_some());
        assert!(parse("definitely not a key").is_none());
        assert!(parse("").is_none());
    }
}

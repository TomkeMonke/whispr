//! whispr - personal dictation.
//!
//! M1 is the cloud-only skeleton: record in the window, transcribe via Groq,
//! read the text back. The global hotkey, overlay and auto-paste arrive in M2;
//! the local Whisper fallback in M3.

mod audio;
mod groq;
mod secrets;
mod settings;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{Manager, State};

use audio::Recorder;
use settings::Settings;

/// Raw captures kept on disk after a transcription, so a failed or surprising
/// result never means the audio is gone.
const KEEP_RECORDINGS: usize = 5;

pub struct AppState {
    recorder: Recorder,
    http: reqwest::Client,
    settings: Mutex<Settings>,
    config_dir: PathBuf,
    recordings_dir: PathBuf,
}

#[derive(Serialize)]
pub struct Transcript {
    text: String,
    duration_secs: f32,
    engine: String,
    /// Where the raw audio was kept, if it could be saved.
    audio_path: Option<String>,
}

#[derive(Serialize)]
pub struct Status {
    recording: bool,
    has_api_key: bool,
}

// ---------------------------------------------------------------------------
// Post-processing seam
// ---------------------------------------------------------------------------

/// Everything between transcription and output passes through here.
///
/// M1 only normalises whitespace. The LLM cleanup pass and custom-vocabulary
/// substitution both land in this function, so adding them later does not
/// change the shape of the pipeline.
///
/// Trailing newlines are stripped deliberately: once M2 pastes into whatever has
/// focus, a trailing newline would submit a terminal prompt the instant the text
/// lands.
fn postprocess(raw: &str) -> String {
    raw.trim().to_string()
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn list_microphones() -> Vec<String> {
    audio::list_input_devices()
}

#[tauri::command]
fn status(state: State<'_, AppState>) -> Status {
    Status {
        recording: state.recorder.is_recording(),
        has_api_key: matches!(secrets::get(), Ok(Some(_))),
    }
}

#[tauri::command]
fn input_level(state: State<'_, AppState>) -> f32 {
    state.recorder.level()
}

#[tauri::command]
fn start_recording(state: State<'_, AppState>) -> Result<(), String> {
    let mic = state.settings.lock().unwrap().microphone.clone();
    state.recorder.start(mic).map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel_recording(state: State<'_, AppState>) {
    state.recorder.cancel();
}

#[tauri::command]
async fn stop_and_transcribe(state: State<'_, AppState>) -> Result<Transcript, String> {
    // Stop and encode first, so the audio is safe on disk before any network work.
    let captured = state.recorder.stop().map_err(|e| e.to_string())?;
    let duration_secs = captured.duration_secs();
    let wav = captured.to_wav_16k().map_err(|e| e.to_string())?;
    let audio_path = persist_recording(&state.recordings_dir, &wav);

    // Snapshot settings and release the lock: the guard must not be held across
    // the await below.
    let (model, language) = {
        let s = state.settings.lock().unwrap();
        (s.model.clone(), s.language.clone())
    };

    let api_key = secrets::get()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| groq::GroqError::MissingKey.to_string())?;

    let raw = groq::transcribe(
        &state.http,
        &api_key,
        wav,
        &model,
        Some(&language),
        // Custom vocabulary plugs in here.
        None,
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(Transcript {
        text: postprocess(&raw),
        duration_secs,
        engine: format!("groq/{model}"),
        audio_path,
    })
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings.lock().unwrap().clone()
}

#[tauri::command]
fn save_settings(state: State<'_, AppState>, settings: Settings) -> Result<(), String> {
    let clean = settings.sanitised();
    settings::save(&state.config_dir, &clean).map_err(|e| e.to_string())?;
    *state.settings.lock().unwrap() = clean;
    Ok(())
}

#[tauri::command]
fn save_api_key(key: String) -> Result<(), String> {
    if key.trim().is_empty() {
        return Err("the key is empty".into());
    }
    secrets::set(&key).map_err(|e| e.to_string())
}

#[tauri::command]
fn clear_api_key() -> Result<(), String> {
    secrets::clear().map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Recording safety net
// ---------------------------------------------------------------------------

/// Keep the raw WAV on disk and prune old ones. Best effort: if this fails the
/// transcription should still go ahead, so it returns an Option rather than a Result.
fn persist_recording(dir: &Path, wav: &[u8]) -> Option<String> {
    std::fs::create_dir_all(dir).ok()?;
    let name = format!("{}.wav", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"));
    let path = dir.join(name);
    std::fs::write(&path, wav).ok()?;
    prune_recordings(dir, KEEP_RECORDINGS);
    Some(path.to_string_lossy().into_owned())
}

fn prune_recordings(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut wavs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .collect();
    if wavs.len() <= keep {
        return;
    }
    // Timestamped names sort chronologically, so this is oldest-first.
    wavs.sort();
    for old in &wavs[..wavs.len() - keep] {
        let _ = std::fs::remove_file(old);
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Must happen before any keychain read.
            if let Err(e) = secrets::init() {
                eprintln!("[whispr] credential store unavailable: {e}");
            }

            let config_dir = app.path().app_config_dir()?;
            let recordings_dir = app.path().app_data_dir()?.join("recordings");
            std::fs::create_dir_all(&config_dir).ok();

            let loaded = settings::load(&config_dir);

            app.manage(AppState {
                recorder: Recorder::new(),
                http: groq::client(),
                settings: Mutex::new(loaded),
                config_dir,
                recordings_dir,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_microphones,
            status,
            input_level,
            start_recording,
            cancel_recording,
            stop_and_transcribe,
            get_settings,
            save_settings,
            save_api_key,
            clear_api_key,
        ])
        .run(tauri::generate_context!())
        .expect("error while running whispr");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postprocess_strips_trailing_newline() {
        // A trailing newline would submit a terminal prompt on paste.
        assert_eq!(postprocess("run the tests\n"), "run the tests");
        assert_eq!(postprocess("  run the tests \r\n"), "run the tests");
    }

    #[test]
    fn postprocess_keeps_internal_structure() {
        assert_eq!(postprocess("line one\n\nline two\n"), "line one\n\nline two");
    }

    #[test]
    fn prune_keeps_the_newest() {
        let dir = std::env::temp_dir().join(format!("whispr-prune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["2026-01-01_00-00-00", "2026-01-02_00-00-00", "2026-01-03_00-00-00"] {
            std::fs::write(dir.join(format!("{name}.wav")), b"x").unwrap();
        }
        prune_recordings(&dir, 2);

        let left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(left.len(), 2);
        assert!(!left.contains(&"2026-01-01_00-00-00.wav".to_string()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prune_leaves_non_wav_files_alone() {
        let dir = std::env::temp_dir().join(format!("whispr-prune2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.wav"), b"x").unwrap();
        std::fs::write(dir.join("b.wav"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        prune_recordings(&dir, 1);
        assert!(dir.join("notes.txt").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}

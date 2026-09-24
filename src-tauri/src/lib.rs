//! whispr - personal dictation.
//!
//! Record in the window, transcribe with Groq or a local Whisper model, read the
//! text back. Either engine can be primary; the other is the fallback. The global
//! hotkey, overlay and auto-paste arrive in M2.

mod audio;
mod groq;
mod local;
mod secrets;
mod settings;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use audio::Recorder;
use settings::{Engine, Settings};

/// Raw captures kept on disk after a transcription, so a failed or surprising
/// result never means the audio is gone.
const KEEP_RECORDINGS: usize = 5;

pub struct AppState {
    recorder: Recorder,
    http: reqwest::Client,
    download_http: reqwest::Client,
    /// Shared with the blocking transcription task, hence the Arc.
    local: Arc<local::Engine>,
    downloading: AtomicBool,
    settings: Mutex<Settings>,
    config_dir: PathBuf,
    recordings_dir: PathBuf,
    models_dir: PathBuf,
}

#[derive(Serialize)]
pub struct Transcript {
    text: String,
    duration_secs: f32,
    engine: String,
    /// Set when the primary engine failed and the other one answered instead.
    fallback_reason: Option<String>,
    /// Where the raw audio was kept, if it could be saved.
    audio_path: Option<String>,
}

#[derive(Serialize)]
pub struct Status {
    recording: bool,
    has_api_key: bool,
    /// The selected local model is on disk.
    has_local_model: bool,
    /// This build runs the local engine on the GPU.
    gpu: bool,
}

#[derive(Clone, Serialize)]
struct DownloadProgress {
    id: String,
    received: u64,
    total: u64,
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
// Routing
// ---------------------------------------------------------------------------

/// The engines to try, in order: the primary first, then the other one, each
/// only if it can actually run. Empty means nothing is set up yet.
fn route(primary: Engine, has_key: bool, has_model: bool) -> Vec<Engine> {
    let order = match primary {
        Engine::Cloud => [Engine::Cloud, Engine::Local],
        Engine::Local => [Engine::Local, Engine::Cloud],
    };
    order
        .into_iter()
        .filter(|e| match e {
            Engine::Cloud => has_key,
            Engine::Local => has_model,
        })
        .collect()
}

const NOTHING_READY: &str =
    "no engine is ready - download a local model or add a Groq API key in settings";

fn local_model_ready(models_dir: &Path, id: &str) -> bool {
    local::find(id).is_some_and(|m| local::is_downloaded(models_dir, m))
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
    let local_model = state.settings.lock().unwrap().local_model.clone();
    Status {
        recording: state.recorder.is_recording(),
        has_api_key: matches!(secrets::get(), Ok(Some(_))),
        has_local_model: local_model_ready(&state.models_dir, &local_model),
        gpu: local::GPU,
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
    // Stop and encode first, so the audio is safe on disk before any engine runs.
    let captured = state.recorder.stop().map_err(|e| e.to_string())?;
    let duration_secs = captured.duration_secs();
    let pcm = captured.to_pcm_16k().map_err(|e| e.to_string())?;
    let wav = audio::encode_wav_16k_mono(&pcm).map_err(|e| e.to_string())?;
    let audio_path = persist_recording(&state.recordings_dir, &wav);

    // Snapshot settings and release the lock: the guard must not be held across
    // the awaits below.
    let s = state.settings.lock().unwrap().clone();

    let api_key = secrets::get().ok().flatten();
    let has_model = local_model_ready(&state.models_dir, &s.local_model);
    let engines = route(s.engine, api_key.is_some(), has_model);
    if engines.is_empty() {
        return Err(NOTHING_READY.into());
    }

    let pcm = Arc::new(pcm);
    let mut wav = Some(wav);
    let mut fallback_reason: Option<String> = None;

    for (i, engine) in engines.iter().enumerate() {
        let is_last = i + 1 == engines.len();
        let attempt = match engine {
            Engine::Cloud => groq::transcribe(
                &state.http,
                api_key.as_deref().unwrap_or_default(),
                // The route holds each engine once, so this take runs at most once.
                wav.take().unwrap_or_default(),
                &s.model,
                Some(&s.language),
                // Custom vocabulary plugs in here.
                None,
            )
            .await
            .map(|raw| (raw, format!("groq/{}", s.model)))
            .map_err(|e| (e.should_fall_back(), e.to_string())),

            Engine::Local => {
                let local = state.local.clone();
                let dir = state.models_dir.clone();
                let pcm = pcm.clone();
                let (model, language) = (s.local_model.clone(), s.language.clone());
                tauri::async_runtime::spawn_blocking(move || {
                    // Custom vocabulary plugs in here too.
                    local.transcribe(&dir, &model, &pcm, &language, None)
                })
                .await
                .map_err(|e| (true, format!("local engine crashed: {e}")))
                .and_then(|r| r.map_err(|e| (true, e.to_string())))
                .map(|raw| {
                    let device = if local::GPU { "gpu" } else { "cpu" };
                    (raw, format!("local/{} ({device})", s.local_model))
                })
            }
        };

        match attempt {
            Ok((raw, engine)) => {
                return Ok(Transcript {
                    text: postprocess(&raw),
                    duration_secs,
                    engine,
                    fallback_reason,
                    audio_path,
                })
            }
            // A bad key or an oversized clip would fail the same way again, and
            // quietly answering from the other engine would hide it.
            Err((retryable, message)) if retryable && !is_last => {
                fallback_reason = Some(message);
            }
            Err((_, message)) => return Err(message),
        }
    }
    Err(NOTHING_READY.into())
}

#[tauri::command]
fn local_models(state: State<'_, AppState>) -> Vec<local::ModelInfo> {
    local::catalog(&state.models_dir)
}

/// Download a local model, emitting `model-download` progress events. Only one
/// download runs at a time.
#[tauri::command]
async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    let spec = local::find(&id).ok_or_else(|| format!("unknown model '{id}'"))?;
    if state.downloading.swap(true, Ordering::SeqCst) {
        return Err("a download is already running".into());
    }
    let result = local::download(&state.download_http, &state.models_dir, spec, |received, total| {
        let _ = app.emit(
            "model-download",
            DownloadProgress {
                id: id.clone(),
                received,
                total,
            },
        );
    })
    .await;
    state.downloading.store(false, Ordering::SeqCst);
    result.map_err(|e| e.to_string())
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

            // whisper.cpp logs every tensor it loads to stderr; route it nowhere.
            whisper_rs::install_logging_hooks();

            let config_dir = app.path().app_config_dir()?;
            let data_dir = app.path().app_data_dir()?;
            let recordings_dir = data_dir.join("recordings");
            let models_dir = data_dir.join("models");
            std::fs::create_dir_all(&config_dir).ok();

            let loaded = settings::load(&config_dir);
            let local = Arc::new(local::Engine::default());

            // Load the local model in the background when it is the primary
            // engine, so the first dictation does not pay for it.
            if loaded.engine == Engine::Local {
                let (local, dir, model) =
                    (local.clone(), models_dir.clone(), loaded.local_model.clone());
                std::thread::spawn(move || local.warm(&dir, &model));
            }

            app.manage(AppState {
                recorder: Recorder::new(),
                http: groq::client(),
                download_http: local::download_client(),
                local,
                downloading: AtomicBool::new(false),
                settings: Mutex::new(loaded),
                config_dir,
                recordings_dir,
                models_dir,
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
            local_models,
            download_model,
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
    fn route_puts_the_primary_first() {
        assert_eq!(route(Engine::Cloud, true, true), [Engine::Cloud, Engine::Local]);
        assert_eq!(route(Engine::Local, true, true), [Engine::Local, Engine::Cloud]);
    }

    #[test]
    fn route_skips_engines_that_cannot_run() {
        // No Groq key: a cloud-primary setup quietly uses local.
        assert_eq!(route(Engine::Cloud, false, true), [Engine::Local]);
        // No model yet: a local-primary setup uses the cloud.
        assert_eq!(route(Engine::Local, true, false), [Engine::Cloud]);
        assert!(route(Engine::Local, false, false).is_empty());
    }

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

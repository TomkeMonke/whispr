//! whispr - personal dictation.
//!
//! Press a global hotkey anywhere, speak, and the transcript is pasted into the
//! focused window. Transcription runs on Groq or a local Whisper model; either
//! can be primary, with the other as the fallback. The window is for settings
//! and for reading back the last result; closing it leaves whispr in the tray.

mod audio;
mod cleanup;
mod groq;
mod hotkey;
mod local;
mod overlay;
mod paste;
mod secrets;
mod settings;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_autostart::ManagerExt;
use tauri_plugin_global_shortcut::GlobalShortcutExt;

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
    /// A transcription is running. Guards against a second one racing it.
    busy: AtomicBool,
    hotkey: hotkey::Hotkey,
    /// Why the hotkey could not be registered, usually another app owning it.
    hotkey_error: Mutex<Option<String>>,
    settings: Mutex<Settings>,
    config_dir: PathBuf,
    recordings_dir: PathBuf,
    models_dir: PathBuf,
}

impl AppState {
    fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Serialize)]
pub struct Transcript {
    /// What gets pasted: cleaned up, when cleanup ran.
    text: String,
    /// The transcript before cleanup, when cleanup changed it.
    raw_text: Option<String>,
    /// Why cleanup was skipped or failed, when it was meant to run.
    cleanup_note: Option<String>,
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
    hotkey_error: Option<String>,
}

/// Emitted as `dictation` for hotkey sessions, so the window can follow along
/// even when the recording was started from another app.
#[derive(Clone, Serialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
enum DictationEvent {
    Recording,
    Transcribing,
    Done {
        transcript: Transcript,
        delivery: Option<paste::Delivery>,
    },
    Error {
        message: String,
    },
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
/// Runs on the raw transcript and again on the cleanup pass's output, so the
/// pasted text gets these guarantees whichever path produced it.
///
/// Trailing newlines are stripped deliberately: the text is pasted into whatever
/// has focus, and a trailing newline would submit a terminal prompt the instant
/// it lands.
///
/// Trailing spaces go too, line by line: the cleanup model writes Markdown-style
/// hard breaks (two spaces before a newline), which would otherwise be pasted.
fn postprocess(raw: &str) -> String {
    raw.trim()
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
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
        hotkey_error: state.hotkey_error.lock().unwrap().clone(),
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
    if state.busy.swap(true, Ordering::SeqCst) {
        return Err("already transcribing".into());
    }
    let result = transcribe_capture(&state).await;
    state.busy.store(false, Ordering::SeqCst);
    result
}

// ---------------------------------------------------------------------------
// Hotkey sessions
// ---------------------------------------------------------------------------

fn start_dictation(app: &AppHandle) {
    let state = app.state::<AppState>();
    let mic = state.settings.lock().unwrap().microphone.clone();
    overlay::show(app);
    let event = match state.recorder.start(mic) {
        Ok(()) => DictationEvent::Recording,
        Err(e) => {
            overlay::hide_after(app, ERROR_LINGER);
            DictationEvent::Error {
                message: e.to_string(),
            }
        }
    };
    let _ = app.emit("dictation", event);
}

/// How long the overlay stays up after a session ends: long enough to read.
const DONE_LINGER: Duration = Duration::from_millis(900);
const ERROR_LINGER: Duration = Duration::from_millis(2500);

async fn finish_dictation(app: AppHandle) {
    let state = app.state::<AppState>();
    if state.busy.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = app.emit("dictation", DictationEvent::Transcribing);
    let result = transcribe_capture(&state).await;
    state.busy.store(false, Ordering::SeqCst);

    let event = match result {
        Ok(transcript) => {
            let delivery = if transcript.text.is_empty() {
                None
            } else {
                let auto_paste = state.settings.lock().unwrap().auto_paste;
                let text = transcript.text.clone();
                // Blocking: waits for modifier keys and the clipboard restore.
                match tauri::async_runtime::spawn_blocking(move || {
                    paste::deliver(&text, auto_paste)
                })
                .await
                {
                    Ok(Ok(delivery)) => Some(delivery),
                    Ok(Err(message)) => {
                        let _ = app.emit("dictation", DictationEvent::Error { message });
                        None
                    }
                    Err(_) => None,
                }
            };
            overlay::hide_after(&app, DONE_LINGER);
            DictationEvent::Done {
                transcript,
                delivery,
            }
        }
        Err(message) => {
            overlay::hide_after(&app, ERROR_LINGER);
            DictationEvent::Error { message }
        }
    };
    let _ = app.emit("dictation", event);
}

/// Register `combo` as the hotkey, replacing any previous one.
fn register_hotkey(app: &AppHandle, combo: &str) -> Result<(), String> {
    let shortcut = hotkey::parse(combo).ok_or_else(|| format!("'{combo}' is not a valid shortcut"))?;
    let shortcuts = app.global_shortcut();
    let _ = shortcuts.unregister_all();
    shortcuts
        .register(shortcut)
        .map_err(|_| format!("{combo} is already taken by another app - pick another"))
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

/// Stop the recorder and run the capture through the engine route.
async fn transcribe_capture(state: &AppState) -> Result<Transcript, String> {
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
    let hint = cleanup::whisper_prompt(&s.vocabulary);

    let mut transcribed: Option<(String, String)> = None;
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
                hint.as_deref(),
            )
            .await
            .map(|raw| (raw, format!("groq/{}", s.model)))
            .map_err(|e| (e.should_fall_back(), e.to_string())),

            Engine::Local => {
                let local = state.local.clone();
                let dir = state.models_dir.clone();
                let pcm = pcm.clone();
                let (model, language) = (s.local_model.clone(), s.language.clone());
                let hint = hint.clone();
                tauri::async_runtime::spawn_blocking(move || {
                    local.transcribe(&dir, &model, &pcm, &language, hint.as_deref())
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
            Ok(done) => {
                transcribed = Some(done);
                break;
            }
            // A bad key or an oversized clip would fail the same way again, and
            // quietly answering from the other engine would hide it.
            Err((retryable, message)) if retryable && !is_last => {
                fallback_reason = Some(message);
            }
            Err((_, message)) => return Err(message),
        }
    }
    let (raw, mut engine) = transcribed.ok_or_else(|| NOTHING_READY.to_string())?;
    let mut text = postprocess(&raw);
    // A lone "." or "..." is Whisper filling a pause, never something said.
    if !text.chars().any(char::is_alphanumeric) {
        text.clear();
    }

    // Cleanup. Skipped when the cloud was just unreachable: it is the same
    // server, and waiting out its timeout would only delay the paste.
    let cloud_failed = s.engine == Engine::Cloud && fallback_reason.is_some();
    let (text, raw_text, cleanup_note) = match api_key.as_deref() {
        Some(key) if s.cleanup && !cloud_failed && cleanup::worth_cleaning(&text) => {
            match cleanup::clean(&state.http, key, &text, &s.vocabulary).await {
                Ok(cleaned) => {
                    engine.push_str(" + cleanup");
                    let cleaned = postprocess(&cleaned);
                    let raw_text = (cleaned != text).then_some(text);
                    (cleaned, raw_text, None)
                }
                Err(e) => (text, None, Some(format!("not cleaned up: {e}"))),
            }
        }
        _ => (text, None, None),
    };

    Ok(Transcript {
        text,
        raw_text,
        cleanup_note,
        duration_secs,
        engine,
        fallback_reason,
        audio_path,
    })
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
fn save_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    settings: Settings,
) -> Result<(), String> {
    // Loading falls back to the default for a bad hotkey; saving says so instead.
    if hotkey::parse(&settings.hotkey).is_none() {
        return Err(format!("'{}' is not a valid shortcut", settings.hotkey.trim()));
    }
    let clean = settings.sanitised();
    let previous_hotkey = state.settings.lock().unwrap().hotkey.clone();
    if clean.hotkey != previous_hotkey {
        if let Err(e) = register_hotkey(&app, &clean.hotkey) {
            // Put the old one back rather than leave no hotkey at all.
            let _ = register_hotkey(&app, &previous_hotkey);
            return Err(e);
        }
        *state.hotkey_error.lock().unwrap() = None;
    }
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

/// Whether whispr starts at login. `None` in a dev build: the registration
/// points at the running executable, which in dev is a throwaway build under
/// target/, so the option only exists in the installed app.
#[tauri::command]
fn get_autostart(app: AppHandle) -> Result<Option<bool>, String> {
    if cfg!(debug_assertions) {
        return Ok(None);
    }
    app.autolaunch()
        .is_enabled()
        .map(Some)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn set_autostart(app: AppHandle, enabled: bool) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("start at login only works in the installed app".into());
    }
    let manager = app.autolaunch();
    if enabled {
        manager.enable()
    } else {
        manager.disable()
    }
    .map_err(|e| e.to_string())
}

/// Start at login is on by default: switch it on the first time the installed
/// app runs. The marker file makes that a one-off, so switching it off in
/// settings sticks.
fn default_autostart_on(app: &AppHandle) {
    if cfg!(debug_assertions) {
        return;
    }
    let Ok(dir) = app.path().app_config_dir() else {
        return;
    };
    let marker = dir.join("autostart-defaulted");
    if marker.exists() {
        return;
    }
    match app.autolaunch().enable() {
        Ok(()) => {
            let _ = std::fs::write(&marker, "");
        }
        Err(e) => eprintln!("[whispr] could not turn on start at login: {e}"),
    }
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

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn build_tray(app: &tauri::App) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "Show whispr", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit whispr", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;

    let mut tray = TrayIconBuilder::with_id("main")
        .tooltip("whispr")
        .menu(&menu)
        // Left click opens the window; the menu is on right click.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show_main_window(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

/// Passed by the login entry, so a start at login goes straight to the tray.
const AUTOSTART_ARG: &str = "--autostart";

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // First, so a second launch exits before anything else starts. It
        // would otherwise fail to register the hotkey the first one holds.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app)
        }))
        .plugin(
            tauri_plugin_autostart::Builder::new()
                .arg(AUTOSTART_ARG)
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| hotkey::handle(app, event))
                .build(),
        )
        // Closing the window hides it: the hotkey has to keep working. Quit is
        // in the tray menu.
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
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

            // Load the local model in the background whenever it will answer
            // first - it is the primary, or there is no key for the cloud - so
            // the first dictation does not pay for it.
            let has_key = matches!(secrets::get(), Ok(Some(_)));
            if loaded.engine == Engine::Local || !has_key {
                let (local, dir, model) =
                    (local.clone(), models_dir.clone(), loaded.local_model.clone());
                std::thread::spawn(move || local.warm(&dir, &model));
            }

            let combo = loaded.hotkey.clone();
            app.manage(AppState {
                recorder: Recorder::new(),
                http: groq::client(),
                download_http: local::download_client(),
                local,
                downloading: AtomicBool::new(false),
                busy: AtomicBool::new(false),
                hotkey: hotkey::Hotkey::default(),
                hotkey_error: Mutex::new(None),
                settings: Mutex::new(loaded),
                config_dir,
                recordings_dir,
                models_dir,
            });

            // A taken hotkey must not stop the app starting: the window still
            // works, and settings shows why the hotkey does not.
            if let Err(e) = register_hotkey(app.handle(), &combo) {
                eprintln!("[whispr] {e}");
                *app.state::<AppState>().hotkey_error.lock().unwrap() = Some(e);
            }

            // Without the overlay, dictation still works - there is just no
            // on-screen sign of it - so a failure here must not stop startup.
            if let Err(e) = overlay::create(app.handle()) {
                eprintln!("[whispr] could not create the overlay: {e}");
            }

            build_tray(app)?;
            default_autostart_on(app.handle());

            // The window starts hidden (tauri.conf.json). A normal launch shows
            // it; a start at login leaves whispr in the tray with the hotkey armed.
            if !std::env::args().any(|a| a == AUTOSTART_ARG) {
                show_main_window(app.handle());
            }
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
            get_autostart,
            set_autostart,
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
    fn postprocess_drops_markdown_hard_breaks() {
        assert_eq!(
            postprocess("1. Fix the login bug.  \n2. Update the copy.  "),
            "1. Fix the login bug.\n2. Update the copy."
        );
        assert_eq!(postprocess("a \r\nb"), "a\nb");
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

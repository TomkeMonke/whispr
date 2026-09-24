//! Local Whisper engine: whisper.cpp via whisper-rs, plus the model download.
//!
//! Needs no key and no network once a model is on disk, which makes it both the
//! offline fallback and a primary engine in its own right. Build with
//! `--features cuda` on a machine with an NVIDIA GPU; without it, whisper.cpp
//! runs on the CPU.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::TARGET_RATE;

const DOWNLOAD_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// whisper.cpp returns no segments at all for input under one second, so short
/// clips are padded with silence up to this length.
const MIN_INPUT_SECS: f32 = 1.2;

pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    file: &'static str,
    size: u64,
    /// SHA-256 of the file, from Hugging Face's LFS pointer. Checked after every
    /// download, so a truncated or tampered file never gets loaded.
    sha256: &'static str,
}

/// English-only (`.en`) models where they exist: whispr pins English, and the
/// `.en` variants beat their multilingual twins at the same size. Turbo has no
/// `.en` build, but is still the most accurate of the three.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "base.en",
        label: "Base - fastest",
        file: "ggml-base.en-q5_1.bin",
        size: 59_721_011,
        sha256: "4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f",
    },
    ModelSpec {
        id: "small.en",
        label: "Small - balanced",
        file: "ggml-small.en-q5_1.bin",
        size: 190_098_681,
        sha256: "bfdff4894dcb76bbf647d56263ea2a96645423f1669176f4844a1bf8e478ad30",
    },
    ModelSpec {
        id: "large-v3-turbo",
        label: "Turbo - most accurate",
        file: "ggml-large-v3-turbo-q5_0.bin",
        size: 574_041_195,
        sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    },
];

/// Whether this binary was built with GPU acceleration.
pub const GPU: bool = cfg!(feature = "cuda");

/// Small on every build. On a GPU it was three times faster than Turbo (0.63 s
/// against 2.0 s for a 6 s clip) and just as exact on clean dictation; on a CPU,
/// Turbo is too slow to dictate with at all. Turbo stays one click away for
/// harder audio.
pub fn default_model() -> &'static str {
    "small.en"
}

pub fn find(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

pub fn model_path(dir: &Path, spec: &ModelSpec) -> PathBuf {
    dir.join(spec.file)
}

/// A size check is enough here: the hash was verified before the file was
/// renamed into place, and a partial file never gets the final name.
pub fn is_downloaded(dir: &Path, spec: &ModelSpec) -> bool {
    std::fs::metadata(model_path(dir, spec)).is_ok_and(|m| m.len() == spec.size)
}

#[derive(Serialize)]
pub struct ModelInfo {
    id: &'static str,
    label: &'static str,
    size_bytes: u64,
    downloaded: bool,
}

pub fn catalog(dir: &Path) -> Vec<ModelInfo> {
    MODELS
        .iter()
        .map(|m| ModelInfo {
            id: m.id,
            label: m.label,
            size_bytes: m.size,
            downloaded: is_downloaded(dir, m),
        })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum LocalError {
    #[error("the local model is not downloaded yet - get it in settings")]
    NoModel,
    #[error("unknown local model '{0}'")]
    UnknownModel(String),
    #[error("model download failed: {0}")]
    Download(String),
    #[error("the downloaded model did not match its checksum - try again")]
    Checksum,
    #[error("could not load the local model: {0}")]
    Load(String),
    #[error("local transcription failed: {0}")]
    Transcribe(String),
    #[error("could not write the model to disk: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

/// A separate client from the Groq one: that one has a 120 s total timeout,
/// which a 550 MB download would blow straight through. This one only gives up
/// when the connection stalls.
pub fn download_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .build()
        .unwrap_or_default()
}

/// Stream the model to `<file>.part`, hashing as it goes, and only rename it into
/// place once size and hash both match.
pub async fn download(
    client: &reqwest::Client,
    dir: &Path,
    spec: &ModelSpec,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<(), LocalError> {
    std::fs::create_dir_all(dir)?;
    let final_path = model_path(dir, spec);
    let part_path = dir.join(format!("{}.part", spec.file));

    let mut response = client
        .get(format!("{DOWNLOAD_BASE}/{}", spec.file))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| LocalError::Download(e.to_string()))?;

    let mut file = std::io::BufWriter::new(std::fs::File::create(&part_path)?);
    let mut hasher = Sha256::new();
    let mut received: u64 = 0;
    let mut last_report: u64 = 0;

    loop {
        let chunk = match response.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(&part_path);
                return Err(LocalError::Download(e.to_string()));
            }
        };
        file.write_all(&chunk)?;
        hasher.update(&chunk);
        received += chunk.len() as u64;
        // Roughly every 1 MB is plenty for a progress bar, and keeps the event
        // stream from flooding the webview.
        if received - last_report >= 1 << 20 {
            on_progress(received, spec.size);
            last_report = received;
        }
    }
    file.flush()?;
    drop(file);

    if received != spec.size || hex(&hasher.finalize()) != spec.sha256 {
        let _ = std::fs::remove_file(&part_path);
        return Err(LocalError::Checksum);
    }
    std::fs::rename(&part_path, &final_path)?;
    on_progress(received, spec.size);
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Holds the loaded model between dictations. Loading takes a second or two
/// (longer for Turbo on a CPU), so paying it once per session matters.
#[derive(Default)]
pub struct Engine {
    loaded: Mutex<Option<(&'static str, Arc<WhisperContext>)>>,
}

impl Engine {
    fn context(&self, dir: &Path, spec: &'static ModelSpec) -> Result<Arc<WhisperContext>, LocalError> {
        let mut loaded = self.loaded.lock().unwrap();
        if let Some((id, ctx)) = loaded.as_ref() {
            if *id == spec.id {
                return Ok(ctx.clone());
            }
        }
        if !is_downloaded(dir, spec) {
            return Err(LocalError::NoModel);
        }
        // Drop the old model before loading the new one, so two never sit in
        // memory (or VRAM) at once.
        *loaded = None;
        let ctx = WhisperContext::new_with_params(
            model_path(dir, spec),
            WhisperContextParameters::default(),
        )
        .map_err(|e| LocalError::Load(e.to_string()))?;
        let ctx = Arc::new(ctx);
        *loaded = Some((spec.id, ctx.clone()));
        Ok(ctx)
    }

    /// Load the model ahead of the first dictation. Best effort.
    pub fn warm(&self, dir: &Path, model_id: &str) {
        if let Some(spec) = find(model_id) {
            let _ = self.context(dir, spec);
        }
    }

    /// Transcribe 16 kHz mono samples. Blocking and CPU-heavy, so call it off
    /// the async runtime.
    ///
    /// `prompt` is Whisper's context hint, the same hook the Groq client takes;
    /// custom vocabulary plugs in there.
    pub fn transcribe(
        &self,
        dir: &Path,
        model_id: &str,
        pcm: &[f32],
        language: &str,
        prompt: Option<&str>,
    ) -> Result<String, LocalError> {
        let spec = find(model_id).ok_or_else(|| LocalError::UnknownModel(model_id.into()))?;
        let ctx = self.context(dir, spec)?;
        let mut state = ctx
            .create_state()
            .map_err(|e| LocalError::Transcribe(e.to_string()))?;

        // Greedy with best_of 1: beam search buys very little on clean dictation
        // and costs several times the decode time on a CPU.
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(threads());
        params.set_language(Some(language));
        params.set_translate(false);
        params.set_no_timestamps(true);
        // Each dictation stands alone; carrying context over only invites the
        // previous clip's words to leak into this one.
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_temperature(0.0);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        if let Some(hint) = prompt.map(str::trim).filter(|h| !h.is_empty()) {
            params.set_initial_prompt(hint);
        }

        let padded = pad_to_min(pcm);
        state
            .full(params, &padded)
            .map_err(|e| LocalError::Transcribe(e.to_string()))?;

        let segments: Vec<String> = state
            .as_iter()
            .filter_map(|seg| seg.to_str_lossy().ok().map(|s| s.into_owned()))
            .collect();
        Ok(join_segments(&segments))
    }
}

/// Leave one core free so the rest of the machine stays responsive while a
/// long clip decodes. whisper.cpp stops scaling past about eight threads.
fn threads() -> i32 {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    cores.saturating_sub(1).clamp(1, 8) as i32
}

fn pad_to_min(pcm: &[f32]) -> Vec<f32> {
    let min = (MIN_INPUT_SECS * TARGET_RATE as f32) as usize;
    let mut out = pcm.to_vec();
    if out.len() < min {
        out.resize(min, 0.0);
    }
    out
}

/// Join segment texts, dropping whisper.cpp's non-speech markers such as
/// `[BLANK_AUDIO]` or `(wind blowing)`. Those would otherwise land in the
/// focused window as if they had been dictated.
fn join_segments(segments: &[String]) -> String {
    segments
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !is_marker(s))
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_marker(s: &str) -> bool {
    (s.starts_with('[') && s.ends_with(']')) || (s.starts_with('(') && s.ends_with(')'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_default_exists() {
        let mut ids: Vec<_> = MODELS.iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), MODELS.len());
        assert!(find(default_model()).is_some());
    }

    #[test]
    fn hashes_are_well_formed() {
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{}", m.id);
            assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        }
    }

    #[test]
    fn hex_matches_a_known_digest() {
        // SHA-256 of the empty string.
        assert_eq!(
            hex(&Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn partial_file_does_not_count_as_downloaded() {
        let dir = std::env::temp_dir().join(format!("whispr-model-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec = &MODELS[0];
        std::fs::write(model_path(&dir, spec), b"truncated").unwrap();
        assert!(!is_downloaded(&dir, spec));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn short_clips_are_padded_long_ones_untouched() {
        assert_eq!(pad_to_min(&[0.5; 1600]).len(), (MIN_INPUT_SECS * 16_000.0) as usize);
        assert_eq!(pad_to_min(&[0.5; 40_000]).len(), 40_000);
    }

    #[test]
    fn non_speech_markers_are_dropped() {
        let segs: Vec<String> = [" Hello there.", "[BLANK_AUDIO]", " (wind blowing) ", " How are you?"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(join_segments(&segs), "Hello there. How are you?");
    }

    #[test]
    fn brackets_inside_speech_survive() {
        let segs = vec!["Call foo(bar) now".to_string()];
        assert_eq!(join_segments(&segs), "Call foo(bar) now");
    }

    /// End to end: downloads a model through the real hash-checked path (cached
    /// under target/models), then transcribes a 16 kHz mono clip of known text.
    ///
    /// `WHISPR_TEST_WAV=clip.wav cargo test --lib -- --ignored transcribes_real_speech --nocapture`
    /// Optional: `WHISPR_TEST_MODEL` (default base.en), `WHISPR_TEST_EXPECT`.
    #[test]
    #[ignore = "downloads a model and needs a speech clip"]
    fn transcribes_real_speech() {
        let wav = std::env::var("WHISPR_TEST_WAV").expect("set WHISPR_TEST_WAV");
        let model = std::env::var("WHISPR_TEST_MODEL").unwrap_or_else(|_| "base.en".into());
        let expect =
            std::env::var("WHISPR_TEST_EXPECT").unwrap_or_else(|_| "quick brown fox".into());

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("models");
        let spec = find(&model).expect("unknown WHISPR_TEST_MODEL");
        if !is_downloaded(&dir, spec) {
            let started = std::time::Instant::now();
            tauri::async_runtime::block_on(download(&download_client(), &dir, spec, |_, _| {}))
                .expect("download");
            eprintln!("downloaded + verified {} in {:.1?}", spec.id, started.elapsed());
        }

        let reader = hound::WavReader::open(&wav).expect("open wav");
        assert_eq!(reader.spec().sample_rate, TARGET_RATE, "clip must be 16 kHz");
        let pcm: Vec<f32> = reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();

        let engine = Engine::default();
        for pass in ["cold", "warm"] {
            let started = std::time::Instant::now();
            let text = engine.transcribe(&dir, &model, &pcm, "en", None).expect("transcribe");
            eprintln!(
                "{pass} ({}, gpu={GPU}): {:.2?} for {:.1}s of audio -> {text:?}",
                spec.id,
                started.elapsed(),
                pcm.len() as f32 / TARGET_RATE as f32,
            );
            assert!(text.to_lowercase().contains(&expect), "got {text:?}");
        }
    }
}

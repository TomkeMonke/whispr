//! Accuracy benchmarks over a folder of real speech: `NN.wav` (16 kHz mono)
//! next to its reference text in `NN.txt`. Ignored by default - the cloud one
//! calls Groq with the saved key, the local one needs a downloaded model.
//!
//! ```text
//! WHISPR_BENCH_DIR=clips WHISPR_BENCH_LANG=pl cargo test --lib -- --ignored bench_groq --nocapture
//! WHISPR_BENCH_DIR=clips WHISPR_BENCH_LANG=pl WHISPR_TEST_MODEL=large-v3-turbo \
//!   WHISPR_MODELS_DIR=%APPDATA%/com.tomek.whispr/models \
//!   cargo test --features cuda --lib -- --ignored bench_local --nocapture
//! ```
//!
//! Optional: `WHISPR_BENCH_MODELS` (Groq models, comma separated, default both),
//! `WHISPR_BENCH_PROMPT` (sent as Whisper's prompt instead of the app's own).
//!
//! WER is word error rate after lowercasing and dropping punctuation. The
//! second figure also ignores diacritics, which shows how many of the errors
//! are only a missing "ł" or "ż".

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{groq, local};

struct Clip {
    name: String,
    wav: Vec<u8>,
    reference: String,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn clips() -> Vec<Clip> {
    let dir = PathBuf::from(env("WHISPR_BENCH_DIR").expect("set WHISPR_BENCH_DIR"));
    let mut wavs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read WHISPR_BENCH_DIR")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .collect();
    wavs.sort();
    wavs.into_iter()
        .filter_map(|wav| {
            let reference = std::fs::read_to_string(wav.with_extension("txt")).ok()?;
            Some(Clip {
                name: wav.file_stem()?.to_string_lossy().into_owned(),
                wav: std::fs::read(&wav).ok()?,
                reference: reference.trim().to_string(),
            })
        })
        .collect()
}

fn pcm(wav: &[u8]) -> Vec<f32> {
    let reader = hound::WavReader::new(std::io::Cursor::new(wav)).expect("read wav");
    assert_eq!(reader.spec().sample_rate, crate::audio::TARGET_RATE, "clips must be 16 kHz");
    reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn words(text: &str, fold_diacritics: bool) -> Vec<String> {
    let lower = text.to_lowercase();
    let mapped: String = lower
        .chars()
        .map(|c| if fold_diacritics { fold(c) } else { c })
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    mapped.split_whitespace().map(str::to_string).collect()
}

fn fold(c: char) -> char {
    match c {
        'ą' => 'a',
        'ć' => 'c',
        'ę' => 'e',
        'ł' => 'l',
        'ń' => 'n',
        'ó' => 'o',
        'ś' => 's',
        'ź' | 'ż' => 'z',
        other => other,
    }
}

/// Word-level edit distance, and the reference length.
fn errors(reference: &str, hypothesis: &str, fold_diacritics: bool) -> (usize, usize) {
    let r = words(reference, fold_diacritics);
    let h = words(hypothesis, fold_diacritics);
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut row = vec![i + 1; h.len() + 1];
        for (j, hw) in h.iter().enumerate() {
            let substitute = prev[j] + usize::from(rw != hw);
            row[j + 1] = substitute.min(prev[j + 1] + 1).min(row[j] + 1);
        }
        prev = row;
    }
    (prev[h.len()], r.len())
}

#[derive(Default)]
struct Score {
    errors: usize,
    folded: usize,
    words: usize,
    elapsed: Duration,
    clips: usize,
}

impl Score {
    fn add(&mut self, clip: &Clip, hypothesis: &str, elapsed: Duration) {
        let (e, n) = errors(&clip.reference, hypothesis, false);
        let (f, _) = errors(&clip.reference, hypothesis, true);
        if e > 0 {
            eprintln!("  {} [{e}/{n}]\n    ref: {}\n    got: {hypothesis}", clip.name, clip.reference);
        }
        self.errors += e;
        self.folded += f;
        self.words += n;
        self.elapsed += elapsed;
        self.clips += 1;
    }

    fn report(&self, label: &str) -> String {
        let pct = |e: usize| 100.0 * e as f32 / self.words.max(1) as f32;
        format!(
            "{label}: WER {:.1}% ({} errors in {} words), ignoring diacritics {:.1}%, {:.2} s per clip",
            pct(self.errors),
            self.errors,
            self.words,
            pct(self.folded),
            self.elapsed.as_secs_f32() / self.clips.max(1) as f32,
        )
    }
}

fn language() -> String {
    env("WHISPR_BENCH_LANG").unwrap_or_else(|| "en".into())
}

/// The app sends Whisper only the vocabulary, which a benchmark has none of.
/// A style hint in Polish ("Dyktuję tekst po polsku, z poprawną interpunkcją")
/// measured no better than none, so there is no per-language prompt.
fn prompt() -> Option<String> {
    env("WHISPR_BENCH_PROMPT")
}

#[test]
#[ignore = "calls the Groq API with the saved key"]
fn bench_groq() {
    crate::secrets::init().expect("credential store");
    let key = crate::secrets::get().expect("read key").expect("no Groq key saved");
    let client = groq::client();
    let clips = clips();
    let language = language();
    let prompt = prompt();
    let models = env("WHISPR_BENCH_MODELS")
        .unwrap_or_else(|| format!("{},{}", groq::MODEL_TURBO, groq::MODEL_LARGE));
    eprintln!("{} clips, language {language}, prompt {prompt:?}", clips.len());

    let mut summary = Vec::new();
    for model in models.split(',').map(str::trim) {
        eprintln!("\n=== {model}");
        let mut score = Score::default();
        for clip in &clips {
            let mut tries = 0;
            let (text, elapsed) = loop {
                let started = Instant::now();
                let result = tauri::async_runtime::block_on(groq::transcribe(
                    &client,
                    &key,
                    clip.wav.clone(),
                    model,
                    Some(&language),
                    prompt.as_deref(),
                ));
                match result {
                    Ok(text) => break (text, started.elapsed()),
                    Err(groq::GroqError::RateLimited) if tries < 5 => {
                        tries += 1;
                        std::thread::sleep(Duration::from_secs(20));
                    }
                    Err(e) => panic!("{}: {e}", clip.name),
                }
            };
            score.add(clip, &text, elapsed);
            // The free tier allows 20 requests a minute.
            std::thread::sleep(Duration::from_millis(3100));
        }
        summary.push(score.report(model));
    }
    eprintln!("\n{}", summary.join("\n"));
}

#[test]
#[ignore = "needs a downloaded local model"]
fn bench_local() {
    let clips = clips();
    let language = language();
    let prompt = prompt();
    let model = env("WHISPR_TEST_MODEL").unwrap_or_else(|| local::default_model().into());
    let dir = env("WHISPR_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("models"));
    eprintln!("{} clips, {model} (gpu={}), language {language}, prompt {prompt:?}", clips.len(), local::GPU);

    let engine = local::Engine::default();
    engine.warm(&dir, &model);
    let mut score = Score::default();
    for clip in &clips {
        let samples = pcm(&clip.wav);
        let started = Instant::now();
        let text = engine
            .transcribe(&dir, &model, &samples, &language, prompt.as_deref())
            .expect("transcribe");
        score.add(clip, &text, started.elapsed());
    }
    eprintln!("\n{}", score.report(&model));
}

#[test]
fn wer_counts_word_edits() {
    assert_eq!(errors("Ala ma kota.", "ala ma kota", false), (0, 3));
    assert_eq!(errors("Ala ma kota", "Ala ma psa i kota", false), (2, 3));
    assert_eq!(errors("żółć", "zolc", false), (1, 1));
    assert_eq!(errors("żółć", "zolc", true), (0, 1));
}

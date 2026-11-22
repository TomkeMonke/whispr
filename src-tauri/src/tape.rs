//! Long recordings to MP3: the microphone, what the computer plays, or both
//! mixed together - a meeting, with both sides of it.
//!
//! Separate from dictation on purpose. Dictation keeps a short clip in memory
//! and hands it to Whisper; this runs for as long as a meeting does, so it
//! mixes and encodes as it goes and only ever holds a fraction of a second.
//! The MP3 is written while recording, so a crash keeps everything up to it.
//!
//! The two sources run on different clocks, and loopback capture goes quiet
//! (no callbacks at all) whenever nothing is playing. So the wall clock sets
//! the pace: every tick the mixer writes out as much audio as time has passed,
//! taking what each source has buffered and filling a gap with silence.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm, Quality, VbrMode};
use serde::Serialize;

use crate::audio::{self, AudioError, Source};

/// The rate everything is mixed and encoded at.
const RATE: u32 = 48_000;

/// Constant bitrate keeps the duration exact in every player without a header
/// that can only be written at the end. 96 kbit/s mono is clear for voices
/// and decent for music: about 43 MB an hour.
const BITRATE: Bitrate = Bitrate::Kbps96;
const KBPS: u64 = 96;

/// How often the mixer wakes up.
const TICK: Duration = Duration::from_millis(50);

/// How far the mixer stays behind the wall clock, so a callback arriving a
/// little late still lands in place instead of leaving a gap.
const LATENCY_SECS: f64 = 0.3;

/// A source more than this far ahead of the mix is trimmed back, so a stall or
/// clock drift can never push it out of sync for the rest of the meeting.
const MAX_BACKLOG_SECS: f64 = 1.0;

#[derive(Debug, thiserror::Error)]
pub enum TapeError {
    #[error("already recording")]
    AlreadyRecording,
    #[error("not recording")]
    NotRecording,
    #[error("turn on the microphone, the computer audio, or both")]
    NothingToRecord,
    #[error("microphone: {0}")]
    Microphone(AudioError),
    #[error("computer audio: {0}")]
    Loopback(AudioError),
    #[error("could not write the recording: {0}")]
    Io(#[from] std::io::Error),
    #[error("mp3 encoder: {0}")]
    Encoder(String),
    #[error("recording thread crashed")]
    ThreadPanic,
}

/// What to record.
#[derive(Debug, Clone)]
pub struct Plan {
    /// `Some(None)` = the default microphone, `None` = no microphone.
    pub microphone: Option<Option<String>>,
    pub system: bool,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct Finished {
    pub path: String,
    pub duration_secs: f64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TapeStatus {
    pub recording: bool,
    pub elapsed_secs: f64,
    pub path: Option<String>,
    pub mic: bool,
    pub system: bool,
}

pub struct Tape {
    active: Mutex<Option<Active>>,
    mic_level: Arc<AtomicU32>,
    system_level: Arc<AtomicU32>,
}

struct Active {
    stop_tx: mpsc::Sender<()>,
    handle: thread::JoinHandle<Result<Finished, TapeError>>,
    started: Instant,
    path: PathBuf,
    mic: bool,
    system: bool,
}

impl Default for Tape {
    fn default() -> Self {
        Self {
            active: Mutex::new(None),
            mic_level: Arc::new(AtomicU32::new(0)),
            system_level: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl Tape {
    pub fn status(&self) -> TapeStatus {
        let active = self.active.lock().unwrap();
        match active.as_ref() {
            Some(a) => TapeStatus {
                recording: true,
                elapsed_secs: a.started.elapsed().as_secs_f64(),
                path: Some(a.path.to_string_lossy().into_owned()),
                mic: a.mic,
                system: a.system,
            },
            None => TapeStatus {
                recording: false,
                elapsed_secs: 0.0,
                path: None,
                mic: false,
                system: false,
            },
        }
    }

    pub fn is_recording(&self) -> bool {
        self.active.lock().unwrap().is_some()
    }

    pub fn mic_level(&self) -> f32 {
        f32::from_bits(self.mic_level.load(Ordering::Relaxed))
    }

    pub fn system_level(&self) -> f32 {
        f32::from_bits(self.system_level.load(Ordering::Relaxed))
    }

    /// Open the sources and start writing. Returns once audio is flowing, so a
    /// missing or busy device is reported here, not as an empty file later.
    pub fn start(&self, plan: Plan) -> Result<(), TapeError> {
        let mut slot = self.active.lock().unwrap();
        if slot.is_some() {
            return Err(TapeError::AlreadyRecording);
        }
        if plan.microphone.is_none() && !plan.system {
            return Err(TapeError::NothingToRecord);
        }

        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), TapeError>>();
        let levels = (Arc::clone(&self.mic_level), Arc::clone(&self.system_level));
        let (mic, system, path) = (plan.microphone.is_some(), plan.system, plan.path.clone());

        let handle = thread::Builder::new()
            .name("whispr-tape".into())
            .spawn(move || run(plan, levels, stop_rx, ready_tx))
            .map_err(|_| TapeError::ThreadPanic)?;

        match ready_rx.recv() {
            Ok(Ok(())) => {
                *slot = Some(Active {
                    stop_tx,
                    handle,
                    started: Instant::now(),
                    path,
                    mic,
                    system,
                });
                Ok(())
            }
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            Err(_) => match handle.join() {
                Ok(Err(e)) => Err(e),
                _ => Err(TapeError::ThreadPanic),
            },
        }
    }

    /// Stop, finish the file and say where it is.
    pub fn stop(&self) -> Result<Finished, TapeError> {
        let active = self
            .active
            .lock()
            .unwrap()
            .take()
            .ok_or(TapeError::NotRecording)?;
        let _ = active.stop_tx.send(());
        let result = active.handle.join().unwrap_or(Err(TapeError::ThreadPanic));
        self.mic_level.store(0f32.to_bits(), Ordering::Relaxed);
        self.system_level.store(0f32.to_bits(), Ordering::Relaxed);
        result
    }
}

/// One live source: its stream, what its callback has delivered, and that
/// audio converted to [`RATE`] and waiting to be mixed.
struct Input {
    // Held only to keep the stream running; dropping it stops capture.
    _stream: cpal::Stream,
    incoming: Arc<Mutex<Vec<f32>>>,
    resampler: Linear,
    queue: VecDeque<f32>,
    level: Arc<AtomicU32>,
}

impl Input {
    fn open(device: &cpal::Device, source: Source, level: Arc<AtomicU32>) -> Result<Self, AudioError> {
        let incoming = Arc::new(Mutex::new(Vec::new()));
        let (stream, rate) =
            audio::open_stream(device, source, Arc::clone(&incoming), Arc::clone(&level))?;
        Ok(Self {
            _stream: stream,
            incoming,
            resampler: Linear::new(rate, RATE),
            queue: VecDeque::new(),
            level,
        })
    }

    /// Move what the callback delivered into the queue, at [`RATE`].
    fn pull(&mut self) {
        let fresh = std::mem::take(&mut *self.incoming.lock().unwrap());
        if fresh.is_empty() {
            // Loopback sends nothing at all while nothing plays; without this
            // the meter would freeze on the last sound.
            self.level.store(0f32.to_bits(), Ordering::Relaxed);
        }
        self.resampler.push(&fresh, &mut self.queue);
    }
}

fn run(
    plan: Plan,
    (mic_level, system_level): (Arc<AtomicU32>, Arc<AtomicU32>),
    stop_rx: mpsc::Receiver<()>,
    ready_tx: mpsc::Sender<Result<(), TapeError>>,
) -> Result<Finished, TapeError> {
    macro_rules! bail {
        ($err:expr) => {{
            let _ = ready_tx.send(Err($err));
            return Err(TapeError::ThreadPanic);
        }};
    }

    let mut inputs = Vec::new();
    if let Some(name) = &plan.microphone {
        let opened = audio::find_input_device(name.as_deref())
            .and_then(|d| Input::open(&d, Source::Microphone, mic_level));
        match opened {
            Ok(input) => inputs.push(input),
            Err(e) => bail!(TapeError::Microphone(e)),
        }
    }
    if plan.system {
        let opened = audio::default_output_device()
            .and_then(|d| Input::open(&d, Source::Loopback, system_level));
        match opened {
            Ok(input) => inputs.push(input),
            Err(e) => bail!(TapeError::Loopback(e)),
        }
    }

    let mut encoder = match new_encoder() {
        Ok(e) => e,
        Err(e) => bail!(e),
    };
    let file = match create_file(&plan.path) {
        Ok(f) => f,
        Err(e) => bail!(TapeError::Io(e)),
    };
    let mut out = BufWriter::new(file);

    let _ = ready_tx.send(Ok(()));
    let started = Instant::now();

    let latency = (LATENCY_SECS * RATE as f64) as u64;
    let max_backlog = (MAX_BACKLOG_SECS * RATE as f64) as usize;
    let mut written: u64 = 0;
    let mut mix: Vec<f32> = Vec::new();
    let mut mp3: Vec<u8> = Vec::new();

    let result = loop {
        let stopping = !matches!(stop_rx.recv_timeout(TICK), Err(mpsc::RecvTimeoutError::Timeout));
        for input in &mut inputs {
            input.pull();
        }

        // On the last pass, everything still queued goes out.
        let due = if stopping {
            inputs.iter().map(|i| i.queue.len()).max().unwrap_or(0)
        } else {
            let now = (started.elapsed().as_secs_f64() * RATE as f64) as u64;
            now.saturating_sub(latency).saturating_sub(written) as usize
        };

        mix_into(&mut mix, &mut inputs, due);
        if !stopping {
            for input in &mut inputs {
                let excess = input.queue.len().saturating_sub(max_backlog);
                input.queue.drain(..excess);
            }
        }
        written += due as u64;

        mp3.clear();
        if let Err(e) = encode(&mut encoder, &mix, &mut mp3, stopping) {
            break Err(e);
        }
        if let Err(e) = out.write_all(&mp3).and_then(|_| out.flush()) {
            break Err(TapeError::Io(e));
        }
        if stopping {
            break Ok(());
        }
    };

    // Stop capturing before the file is closed.
    drop(inputs);
    drop(out);
    result?;

    let bytes = std::fs::metadata(&plan.path).map(|m| m.len()).unwrap_or(0);
    Ok(Finished {
        path: plan.path.to_string_lossy().into_owned(),
        duration_secs: written as f64 / RATE as f64,
        bytes,
    })
}

/// Sum the next `due` samples of every input into `mix`, with silence where an
/// input has nothing (loopback while nothing plays).
fn mix_into(mix: &mut Vec<f32>, inputs: &mut [Input], due: usize) {
    mix.clear();
    mix.resize(due, 0.0);
    for input in inputs {
        for (slot, sample) in mix.iter_mut().zip(input.queue.drain(..due.min(input.queue.len()))) {
            *slot += sample;
        }
    }
    for s in mix.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }
}

fn new_encoder() -> Result<mp3lame_encoder::Encoder, TapeError> {
    let err = |e: mp3lame_encoder::BuildError| TapeError::Encoder(e.to_string());
    let mut b = Builder::new().ok_or_else(|| TapeError::Encoder("could not start LAME".into()))?;
    b.set_num_channels(1).map_err(err)?;
    b.set_sample_rate(RATE).map_err(err)?;
    b.set_brate(BITRATE).map_err(err)?;
    b.set_vbr_mode(VbrMode::Off).map_err(err)?;
    // The Xing header can only be filled in at the end; CBR does not need it.
    b.set_to_write_vbr_tag(false).map_err(err)?;
    b.set_quality(Quality::Good).map_err(err)?;
    b.build().map_err(err)
}

fn encode(
    encoder: &mut mp3lame_encoder::Encoder,
    pcm: &[f32],
    out: &mut Vec<u8>,
    last: bool,
) -> Result<(), TapeError> {
    let err = |e: mp3lame_encoder::EncodeError| TapeError::Encoder(e.to_string());
    out.reserve(mp3lame_encoder::max_required_buffer_size(pcm.len()));
    if !pcm.is_empty() {
        encoder.encode_to_vec(MonoPcm(pcm), out).map_err(err)?;
    }
    if last {
        // A flush needs up to 7200 bytes.
        out.reserve(7200);
        encoder.flush_to_vec::<FlushNoGap>(out).map_err(err)?;
    }
    Ok(())
}

fn create_file(path: &Path) -> std::io::Result<File> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    File::options().write(true).create_new(true).open(path)
}

/// A free file name in `dir` for a recording started now, such as
/// "2026-11-22 14-30-05.mp3".
pub fn next_path(dir: &Path) -> PathBuf {
    let stamp = chrono::Local::now().format("%Y-%m-%d %H-%M-%S").to_string();
    let mut path = dir.join(format!("{stamp}.mp3"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stamp} ({n}).mp3"));
        n += 1;
    }
    path
}

/// A recording on disk, for the list in the window.
#[derive(Debug, Clone, Serialize)]
pub struct Saved {
    pub name: String,
    pub path: String,
    pub bytes: u64,
    /// From the size: exact for whispr's own constant-bitrate files.
    pub duration_secs: f64,
}

/// The newest recordings in `dir`, newest first.
pub fn list(dir: &Path, limit: usize) -> Vec<Saved> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<(std::time::SystemTime, Saved)> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("mp3")))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            let modified = meta.modified().ok()?;
            let bytes = meta.len();
            Some((
                modified,
                Saved {
                    name: e.file_name().to_string_lossy().into_owned(),
                    path: e.path().to_string_lossy().into_owned(),
                    bytes,
                    duration_secs: bytes as f64 * 8.0 / (KBPS as f64 * 1000.0),
                },
            ))
        })
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0));
    files.into_iter().take(limit).map(|(_, s)| s).collect()
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Streaming linear-interpolation resampler. Audio arrives in callback-sized
/// pieces, so it carries the last sample and its position across calls. Good
/// enough between the usual 44.1 and 48 kHz device rates; dictation's sinc
/// resampler works on whole clips and would need the full clip here.
struct Linear {
    /// Input samples per output sample.
    step: f64,
    /// Position of the next output sample, between `prev` (0.0) and the next
    /// input sample (1.0).
    pos: f64,
    prev: f32,
}

impl Linear {
    fn new(from: u32, to: u32) -> Self {
        Self {
            step: from as f64 / to as f64,
            pos: 0.0,
            prev: 0.0,
        }
    }

    fn push(&mut self, input: &[f32], out: &mut VecDeque<f32>) {
        for &next in input {
            while self.pos < 1.0 {
                let t = self.pos as f32;
                out.push_back(self.prev + (next - self.prev) * t);
                self.pos += self.step;
            }
            self.pos -= 1.0;
            self.prev = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
        (0..(rate as f64 * secs) as usize)
            .map(|i| (std::f64::consts::TAU * freq * i as f64 / rate as f64).sin() as f32 * 0.5)
            .collect()
    }

    #[test]
    fn resampler_keeps_the_length_in_ratio() {
        for from in [16_000, 44_100, 48_000, 96_000] {
            let mut r = Linear::new(from, RATE);
            let mut out = VecDeque::new();
            // In callback-sized pieces, as the streams deliver it.
            for piece in tone(440.0, from, 2.0).chunks(441) {
                r.push(piece, &mut out);
            }
            let expected = 2 * RATE as i64;
            assert!(
                (out.len() as i64 - expected).abs() <= 2,
                "{from} Hz: {} samples, expected {expected}",
                out.len()
            );
        }
    }

    #[test]
    fn resampler_is_identity_at_the_same_rate() {
        let input = tone(440.0, RATE, 0.1);
        let mut r = Linear::new(RATE, RATE);
        let mut out = VecDeque::new();
        r.push(&input, &mut out);
        // One sample late: the first output is the starting zero.
        let out: Vec<f32> = out.into_iter().collect();
        assert_eq!(&out[1..], &input[..input.len() - 1]);
    }

    #[test]
    fn mp3_encodes_and_flushes() {
        let mut enc = new_encoder().unwrap();
        let mut mp3 = Vec::new();
        for chunk in tone(440.0, RATE, 1.0).chunks(2400) {
            encode(&mut enc, chunk, &mut mp3, false).unwrap();
        }
        encode(&mut enc, &[], &mut mp3, true).unwrap();
        // MPEG-1 Layer III frame sync, then about a second at 96 kbit/s.
        assert_eq!(mp3[0], 0xFF);
        assert_eq!(mp3[1] & 0xE0, 0xE0);
        let expected = (KBPS * 1000 / 8) as i64;
        assert!(
            (mp3.len() as i64 - expected).abs() < expected / 10,
            "{} bytes for 1 s",
            mp3.len()
        );
    }

    #[test]
    fn next_path_never_overwrites() {
        let dir = std::env::temp_dir().join(format!("whispr-tape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let first = next_path(&dir);
        std::fs::write(&first, b"x").unwrap();
        let second = next_path(&dir);
        assert_ne!(first, second);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Records 5 seconds of the default mic plus whatever the computer plays.
    /// `cargo test --lib -- --ignored record_mic_and_system --nocapture`
    #[test]
    #[ignore = "needs audio devices"]
    fn record_mic_and_system() {
        let dir = std::env::temp_dir().join("whispr-tape-live");
        let path = next_path(&dir);
        let tape = Tape::default();
        tape.start(Plan {
            microphone: Some(None),
            system: true,
            path,
        })
        .expect("could not start");
        let mut peaks = (0f32, 0f32);
        for _ in 0..50 {
            std::thread::sleep(Duration::from_millis(100));
            peaks.0 = peaks.0.max(tape.mic_level());
            peaks.1 = peaks.1.max(tape.system_level());
        }
        let done = tape.stop().expect("could not stop");
        println!(
            "{} - {:.2} s, {} bytes, mic peak {:.3}, system peak {:.3}",
            done.path, done.duration_secs, done.bytes, peaks.0, peaks.1
        );
        assert!((done.duration_secs - 5.0).abs() < 0.5);
    }
}

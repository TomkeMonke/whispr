//! Microphone capture and signal conditioning.
//!
//! Capture runs on a dedicated OS thread because `cpal::Stream` is `!Send` under
//! WASAPI, so it can never live in Tauri's managed state. The thread owns the
//! stream and parks on a stop channel; everything else only touches the shared
//! sample buffer and the level meter.
//!
//! Capturing in Rust rather than the webview is deliberate: it keeps working when
//! the window is hidden (which M2's global hotkey depends on) and sidesteps the
//! WebView2 microphone permission prompt entirely.

use std::io::Cursor;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

/// Whisper wants 16 kHz mono, both locally and at Groq. We convert once on
/// capture so the cloud and local paths share a single buffer.
pub const TARGET_RATE: u32 = 16_000;

/// Clips shorter than this are almost always a mis-trigger, not speech.
const MIN_DURATION_SECS: f32 = 0.25;

/// Peak below this across the whole clip means the mic captured nothing useful.
///
/// Measured noise floor on a quiet room with the built-in array mic was 0.0037,
/// and speech peaks land around 0.1-0.5, so this sits roughly 4x above the floor
/// and well under an order of magnitude below even a soft talker. Set any closer
/// to the floor and silent clips start slipping through into paid API calls.
const SILENCE_PEAK: f32 = 0.015;

/// Frame length for the speech check: short enough that a key click fills at
/// most a couple of frames, long enough to smooth over single samples.
const FRAME_SECS: f32 = 0.02;

/// A frame counts as voiced when its RMS is this many times the clip's own
/// noise floor (its quietest 10% of frames), clamped to a sane range: the
/// ceiling keeps a clip with no pauses in it (all speech, so the "floor" is
/// speech too) from gating itself out.
const VOICED_OVER_FLOOR: f32 = 3.0;
const MIN_VOICED_RMS: f32 = 0.01;
const MAX_VOICED_RMS: f32 = 0.03;

/// Longest unbroken voiced stretch a clip needs to count as speech. Even "yes"
/// holds its vowel for 200+ ms; clicks, taps and bumps last 20-40 ms. Measured
/// on the desktop mic: an empty clip peaked at 20 ms, the shortest real phrase
/// at 420 ms.
///
/// Whisper never returns nothing: fed an empty clip it invents a stock line
/// ("Thank you.", "you", "."), so the clip has to be stopped before it.
const MIN_VOICED_RUN_SECS: f32 = 0.1;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("no microphone available")]
    NoDevice,
    #[error("microphone '{0}' not found")]
    UnknownDevice(String),
    #[error("unsupported sample format: {0:?}")]
    UnsupportedFormat(SampleFormat),
    #[error("already recording")]
    AlreadyRecording,
    #[error("not recording")]
    NotRecording,
    #[error("recording too short - hold it a moment longer")]
    TooShort,
    #[error("nothing was recorded - check the microphone is not muted")]
    Silent,
    #[error("no speech heard")]
    NoSpeech,
    #[error("capture thread panicked")]
    ThreadPanic,
    /// Startup failed and the real error was already handed to `start` over the
    /// ready channel. Never surfaces to callers.
    #[error("capture aborted during startup")]
    Aborted,
    #[error("audio device error: {0}")]
    Device(#[from] cpal::Error),
    #[error("wav encoding failed: {0}")]
    Wav(#[from] hound::Error),
}

/// A finished capture, already downmixed to mono but still at the device rate.
pub struct Captured {
    pub mono: Vec<f32>,
    pub sample_rate: u32,
}

impl Captured {
    pub fn duration_secs(&self) -> f32 {
        self.mono.len() as f32 / self.sample_rate as f32
    }

    fn peak(&self) -> f32 {
        self.mono.iter().fold(0.0f32, |m, s| m.max(s.abs()))
    }

    /// Gate out mis-triggers and silence, then resample to 16 kHz. The local
    /// engine takes these samples as-is; the cloud path encodes them to WAV.
    pub fn to_pcm_16k(&self) -> Result<Vec<f32>, AudioError> {
        if self.duration_secs() < MIN_DURATION_SECS {
            return Err(AudioError::TooShort);
        }
        if self.peak() < SILENCE_PEAK {
            return Err(AudioError::Silent);
        }
        if longest_voiced_run_secs(&self.mono, self.sample_rate) < MIN_VOICED_RUN_SECS {
            return Err(AudioError::NoSpeech);
        }
        Ok(resample_mono(&self.mono, self.sample_rate, TARGET_RATE))
    }
}

/// The longest unbroken stretch of frames that stand clear of the clip's own
/// noise floor. Relative to the floor, so a noisy mic and a quiet one both work.
fn longest_voiced_run_secs(mono: &[f32], rate: u32) -> f32 {
    let frame = ((rate as f32 * FRAME_SECS) as usize).max(1);
    let rms: Vec<f32> = mono
        .chunks_exact(frame)
        .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / frame as f32).sqrt())
        .collect();
    if rms.is_empty() {
        return 0.0;
    }
    let mut sorted = rms.clone();
    sorted.sort_by(f32::total_cmp);
    let floor = sorted[sorted.len() / 10];
    let threshold = (floor * VOICED_OVER_FLOOR).clamp(MIN_VOICED_RMS, MAX_VOICED_RMS);

    let (mut run, mut best) = (0usize, 0usize);
    for level in rms {
        run = if level > threshold { run + 1 } else { 0 };
        best = best.max(run);
    }
    best as f32 * frame as f32 / rate as f32
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

pub struct Recorder {
    active: Mutex<Option<Active>>,
    level: Arc<AtomicU32>,
}

struct Active {
    stop_tx: mpsc::Sender<()>,
    handle: thread::JoinHandle<Result<Captured, AudioError>>,
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            active: Mutex::new(None),
            level: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn is_recording(&self) -> bool {
        self.active.lock().unwrap().is_some()
    }

    /// Current input level, 0.0..=1.0. Polled by the UI meter.
    pub fn level(&self) -> f32 {
        f32::from_bits(self.level.load(Ordering::Relaxed))
    }

    /// Open the mic and start capturing.
    ///
    /// Returns only once the stream is actually running, so a busy or blocked
    /// microphone surfaces here rather than as an empty recording later.
    pub fn start(&self, device_name: Option<String>) -> Result<(), AudioError> {
        let mut slot = self.active.lock().unwrap();
        if slot.is_some() {
            return Err(AudioError::AlreadyRecording);
        }

        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), AudioError>>();
        let level = Arc::clone(&self.level);
        self.level.store(0f32.to_bits(), Ordering::Relaxed);

        let handle = thread::Builder::new()
            .name("whispr-capture".into())
            .spawn(move || run_capture(device_name, stop_rx, level, ready_tx))
            .map_err(|_| AudioError::ThreadPanic)?;

        // Wait for the stream to come up (or fail) before reporting success.
        match ready_rx.recv() {
            Ok(Ok(())) => {
                *slot = Some(Active { stop_tx, handle });
                Ok(())
            }
            // The thread reported the failure over the channel and returned
            // `Aborted`; the error that matters is the one we just received.
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            // Sender dropped without a verdict: the thread died early.
            Err(_) => match handle.join() {
                Ok(Err(e)) => Err(e),
                _ => Err(AudioError::ThreadPanic),
            },
        }
    }

    /// Stop capturing and hand back everything recorded.
    pub fn stop(&self) -> Result<Captured, AudioError> {
        let active = self
            .active
            .lock()
            .unwrap()
            .take()
            .ok_or(AudioError::NotRecording)?;

        // If the thread already exited the send fails, which is fine - the join
        // below still yields whatever it captured.
        let _ = active.stop_tx.send(());
        self.level.store(0f32.to_bits(), Ordering::Relaxed);

        match active.handle.join() {
            Ok(result) => result,
            Err(_) => Err(AudioError::ThreadPanic),
        }
    }

    /// Stop and discard. Used when a capture is abandoned.
    pub fn cancel(&self) {
        if let Some(active) = self.active.lock().unwrap().take() {
            let _ = active.stop_tx.send(());
            let _ = active.handle.join();
        }
        self.level.store(0f32.to_bits(), Ordering::Relaxed);
    }
}

fn run_capture(
    device_name: Option<String>,
    stop_rx: mpsc::Receiver<()>,
    level: Arc<AtomicU32>,
    ready_tx: mpsc::Sender<Result<(), AudioError>>,
) -> Result<Captured, AudioError> {
    // Anything that fails before the stream is live gets handed to `start`
    // through ready_tx, so it can be returned synchronously.
    macro_rules! bail {
        ($err:expr) => {{
            let _ = ready_tx.send(Err($err));
            return Err(AudioError::Aborted);
        }};
    }

    let host = cpal::default_host();

    let device = match &device_name {
        Some(name) => {
            let found = host
                .input_devices()
                .ok()
                .and_then(|mut it| it.find(|d| d.to_string() == *name));
            match found {
                Some(d) => d,
                None => bail!(AudioError::UnknownDevice(name.clone())),
            }
        }
        None => match host.default_input_device() {
            Some(d) => d,
            None => bail!(AudioError::NoDevice),
        },
    };

    let supported = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => bail!(AudioError::Device(e)),
    };

    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let format = supported.sample_format();
    let config: StreamConfig = supported.config();

    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));

    let stream = {
        let buffer = Arc::clone(&buffer);
        let level = Arc::clone(&level);
        let built = match format {
            SampleFormat::F32 => build_stream::<f32>(&device, config, channels, buffer, level, |s| s),
            SampleFormat::I16 => build_stream::<i16>(&device, config, channels, buffer, level, |s| {
                s as f32 / -(i16::MIN as f32)
            }),
            SampleFormat::I32 => build_stream::<i32>(&device, config, channels, buffer, level, |s| {
                s as f32 / -(i32::MIN as f32)
            }),
            SampleFormat::I8 => build_stream::<i8>(&device, config, channels, buffer, level, |s| {
                s as f32 / -(i8::MIN as f32)
            }),
            SampleFormat::U8 => build_stream::<u8>(&device, config, channels, buffer, level, |s| {
                (s as f32 - 128.0) / 128.0
            }),
            other => bail!(AudioError::UnsupportedFormat(other)),
        };
        match built {
            Ok(s) => s,
            Err(e) => bail!(AudioError::Device(e)),
        }
    };

    if let Err(e) = stream.play() {
        bail!(AudioError::Device(e));
    }

    // Live from here on.
    let _ = ready_tx.send(Ok(()));

    // Park until stop() rings, or forever if the sender vanished.
    let _ = stop_rx.recv();

    drop(stream);

    let mono = std::mem::take(&mut *buffer.lock().unwrap());
    Ok(Captured { mono, sample_rate })
}

fn build_stream<T>(
    device: &cpal::Device,
    config: StreamConfig,
    channels: usize,
    buffer: Arc<Mutex<Vec<f32>>>,
    level: Arc<AtomicU32>,
    convert: fn(T) -> f32,
) -> Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample + Send + 'static,
{
    device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _| {
            // Downmix to mono here so we never store N channels we don't want.
            let frames = data.len() / channels.max(1);
            let mut peak = 0.0f32;
            let mut out = Vec::with_capacity(frames);
            for frame in data.chunks_exact(channels.max(1)) {
                let mut acc = 0.0f32;
                for &s in frame {
                    acc += convert(s);
                }
                let v = acc / channels.max(1) as f32;
                peak = peak.max(v.abs());
                out.push(v);
            }
            level.store(peak.to_bits(), Ordering::Relaxed);
            if let Ok(mut buf) = buffer.lock() {
                buf.extend_from_slice(&out);
            }
        },
        move |err| {
            eprintln!("[whispr] input stream error: {err}");
        },
        None,
    )
}

/// Microphone names, for the settings dropdown. Best-effort: a device that
/// errors while being described is simply skipped.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let default = host.default_input_device().map(|d| d.to_string());
    let mut names: Vec<String> = host
        .input_devices()
        .map(|it| it.map(|d| d.to_string()).collect())
        .unwrap_or_default();
    names.dedup();
    // Surface the system default first.
    if let Some(def) = default {
        if let Some(pos) = names.iter().position(|n| *n == def) {
            names.swap(0, pos);
        }
    }
    names
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Half-width of the sinc kernel, in output-side samples.
const KERNEL_HALF: usize = 16;

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let p = std::f64::consts::PI * x;
        p.sin() / p
    }
}

/// Blackman window over the normalised range [-1, 1].
fn blackman(x: f64) -> f64 {
    if !(-1.0..1.0).contains(&x) {
        return 0.0;
    }
    let n = (x + 1.0) * 0.5;
    let tau = std::f64::consts::TAU;
    0.42 - 0.5 * (tau * n).cos() + 0.08 * (2.0 * tau * n).cos()
}

/// Convert mono audio between sample rates with a windowed-sinc kernel.
///
/// When downsampling, the kernel cutoff drops to the output Nyquist and the
/// kernel widens to match, so content above the new Nyquist is attenuated
/// instead of aliasing back into the speech band. Coefficients are normalised
/// per output sample, which holds DC gain at unity and keeps the clip edges
/// well behaved.
pub fn resample_mono(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;

    // Cutoff in cycles per *input* sample.
    let cutoff = if ratio < 1.0 { 0.5 * ratio } else { 0.5 };
    // Widen the kernel by the same factor the cutoff narrowed by.
    let half = (KERNEL_HALF as f64 / ratio.min(1.0)).ceil() as isize;

    let mut out = Vec::with_capacity(out_len);
    for t in 0..out_len {
        let center = t as f64 / ratio;
        let base = center.floor() as isize;

        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for k in (base - half + 1)..=(base + half) {
            let dist = center - k as f64;
            let w = blackman(dist / half as f64);
            if w == 0.0 {
                continue;
            }
            let coeff = w * sinc(2.0 * cutoff * dist);
            // Treat out-of-range taps as silence but still let them pull the
            // normaliser, so edges fade rather than ring.
            if k >= 0 && (k as usize) < input.len() {
                acc += coeff * input[k as usize] as f64;
            }
            norm += coeff;
        }

        out.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        });
    }
    out
}

/// Encode mono f32 samples as 16-bit PCM WAV at [`TARGET_RATE`].
pub fn encode_wav_16k_mono(samples: &[f32]) -> Result<Vec<u8>, AudioError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            // Scale by 32767 so +1.0 maps inside range rather than wrapping.
            writer.write_sample((clamped * i16::MAX as f32).round() as i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
        let n = (rate as f64 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / rate as f64;
                (std::f64::consts::TAU * freq * t).sin() as f32
            })
            .collect()
    }

    /// RMS over the middle 80%, so kernel edge effects don't skew the result.
    fn inner_rms(x: &[f32]) -> f32 {
        let lo = x.len() / 10;
        let hi = x.len() - x.len() / 10;
        let slice = &x[lo..hi];
        (slice.iter().map(|v| (v * v) as f64).sum::<f64>() / slice.len() as f64).sqrt() as f32
    }

    /// Room hiss at `level`, deterministic so the tests are stable.
    fn hiss(rate: u32, secs: f64, level: f32) -> Vec<f32> {
        let mut x: u32 = 12345;
        (0..(rate as f64 * secs) as usize)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * level
            })
            .collect()
    }

    fn captured(mono: Vec<f32>, rate: u32) -> Captured {
        Captured { mono, sample_rate: rate }
    }

    #[test]
    fn clicks_in_a_quiet_room_are_not_speech() {
        let rate = 48_000;
        let mut mono = hiss(rate, 1.5, 0.01);
        // Two key clicks: loud, but 5 ms each.
        for start in [20_000, 50_000] {
            for s in &mut mono[start..start + 240] {
                *s = 0.4;
            }
        }
        assert!(matches!(
            captured(mono, rate).to_pcm_16k(),
            Err(AudioError::NoSpeech)
        ));
    }

    #[test]
    fn a_short_word_over_hiss_is_speech() {
        let rate = 48_000;
        let mut mono = hiss(rate, 1.5, 0.01);
        // A 250 ms vowel, about a quiet "yes".
        let word = tone(220.0, rate, 0.25);
        for (s, w) in mono[24_000..].iter_mut().zip(word) {
            *s += w * 0.08;
        }
        assert!(captured(mono, rate).to_pcm_16k().is_ok());
    }

    #[test]
    fn a_clip_that_is_all_speech_still_counts() {
        // No quiet frames, so the floor is the speech itself.
        let rate = 16_000;
        let mono: Vec<f32> = tone(200.0, rate, 1.0).into_iter().map(|v| v * 0.3).collect();
        assert!(longest_voiced_run_secs(&mono, rate) > 0.5);
    }

    #[test]
    fn same_rate_is_identity() {
        let input = tone(440.0, 16_000, 0.1);
        assert_eq!(resample_mono(&input, 16_000, 16_000), input);
    }

    #[test]
    fn output_length_follows_ratio() {
        let input = tone(440.0, 48_000, 1.0);
        let out = resample_mono(&input, 48_000, 16_000);
        assert!(
            (out.len() as i64 - 16_000).abs() <= 1,
            "expected ~16000 samples, got {}",
            out.len()
        );

        let input = tone(440.0, 44_100, 1.0);
        let out = resample_mono(&input, 44_100, 16_000);
        assert!(
            (out.len() as i64 - 16_000).abs() <= 1,
            "expected ~16000 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn preserves_dc() {
        let input = vec![0.5f32; 48_000];
        let out = resample_mono(&input, 48_000, 16_000);
        for (i, v) in out.iter().enumerate().skip(200).take(out.len() - 400) {
            assert!(
                (v - 0.5).abs() < 1e-3,
                "sample {i} drifted from DC: {v}"
            );
        }
    }

    #[test]
    fn preserves_in_band_tone() {
        // 440 Hz is far below the 8 kHz output Nyquist; amplitude should survive.
        let input = tone(440.0, 48_000, 0.5);
        let out = resample_mono(&input, 48_000, 16_000);
        let expected = std::f32::consts::FRAC_1_SQRT_2; // RMS of a unit sine
        let got = inner_rms(&out);
        assert!(
            (got - expected).abs() < 0.02,
            "expected RMS ~{expected}, got {got}"
        );
    }

    #[test]
    fn rejects_out_of_band_tone_instead_of_aliasing() {
        // 12 kHz at 48 kHz is above the 8 kHz Nyquist of a 16 kHz stream. Without
        // an anti-alias filter this folds down to 4 kHz at full amplitude, landing
        // squarely in the speech band. It must be attenuated instead.
        let input = tone(12_000.0, 48_000, 0.5);
        let out = resample_mono(&input, 48_000, 16_000);
        let got = inner_rms(&out);
        assert!(
            got < 0.02,
            "12 kHz tone should be filtered out, but RMS was {got}"
        );
    }

    #[test]
    fn upsampling_works_too() {
        let input = tone(440.0, 8_000, 0.5);
        let out = resample_mono(&input, 8_000, 16_000);
        assert!((out.len() as i64 - 8_000).abs() <= 1);
        let got = inner_rms(&out);
        assert!(
            (got - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02,
            "upsampled RMS was {got}"
        );
    }

    #[test]
    fn wav_header_is_16k_mono_16bit() {
        let wav = encode_wav_16k_mono(&tone(440.0, 16_000, 0.1)).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");

        let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, TARGET_RATE);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(reader.duration(), 1_600);
    }

    #[test]
    fn short_clip_is_rejected() {
        let captured = Captured {
            mono: tone(440.0, 48_000, 0.1),
            sample_rate: 48_000,
        };
        assert!(matches!(
            captured.to_pcm_16k(),
            Err(AudioError::TooShort)
        ));
    }

    /// Opens the real default microphone, so it is not part of the normal run.
    /// `cargo test --lib -- --ignored capture_from_real_microphone --nocapture`
    #[test]
    #[ignore = "needs a microphone"]
    fn capture_from_real_microphone() {
        use std::io::Write;

        let devices = list_input_devices();
        println!("input devices ({}):", devices.len());
        for (i, d) in devices.iter().enumerate() {
            println!("  {}{}", if i == 0 { "* " } else { "  " }, d);
        }
        println!("  (* = system default, which is the one being used)\n");

        let recorder = Recorder::new();
        recorder
            .start(None)
            .expect("could not open the default input device");

        // Give the tester time to actually start talking.
        for n in (1..=3).rev() {
            print!("\rstarting in {n}... ");
            std::io::stdout().flush().ok();
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        println!("\rSPEAK NOW - recording 3 seconds");
        std::io::stdout().flush().ok();

        // Drop the countdown audio so only the spoken part is measured.
        let _ = recorder.stop();
        recorder.start(None).expect("could not reopen the device");

        std::thread::sleep(std::time::Duration::from_millis(3000));
        let level_while_live = recorder.level();
        println!("done");

        let captured = recorder.stop().expect("stop failed");
        println!(
            "captured {} samples @ {} Hz = {:.2}s, peak {:.4}, level at stop {:.4}",
            captured.mono.len(),
            captured.sample_rate,
            captured.duration_secs(),
            captured.peak(),
            level_while_live,
        );

        assert!(!captured.mono.is_empty(), "no samples arrived from the device");
        assert!(
            captured.duration_secs() > 2.0,
            "expected ~3s, got {:.2}s - callbacks are not firing",
            captured.duration_secs()
        );

        match captured.to_pcm_16k().and_then(|pcm| encode_wav_16k_mono(&pcm)) {
            Ok(wav) => println!("encoded {} bytes of 16 kHz mono wav", wav.len()),
            Err(AudioError::Silent) => {
                println!("device works but captured silence - check the mic is not muted")
            }
            Err(e) => panic!("wav encoding failed: {e}"),
        }
        assert!(!recorder.is_recording(), "recorder did not return to idle");
    }

    #[test]
    fn silent_clip_is_rejected() {
        let captured = Captured {
            mono: vec![0.0; 48_000],
            sample_rate: 48_000,
        };
        assert!(matches!(captured.to_pcm_16k(), Err(AudioError::Silent)));
    }
}

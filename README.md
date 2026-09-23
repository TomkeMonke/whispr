# whispr

Personal dictation app. Tauri, with Groq Whisper in the cloud and local Whisper
as the offline fallback. Windows and macOS.

## Status

**M1 - cloud-only skeleton.** Record in the window, transcribe via Groq, read the
text back.

- [x] M1 - window app, cpal capture, Groq transcription
- [ ] M2 - global hotkey, overlay, auto-paste, tray
- [ ] M3 - local Whisper fallback, model download, offline detection
- [ ] Later - LLM cleanup pass, custom vocabulary
- [ ] Later - macOS pass

## Prerequisites

- Node 20+
- Rust stable (`rustup default stable`)
- **Windows:** MSVC C++ Build Tools. Rust installs fine without them but cannot
  link a binary - not even `cargo check`, since build scripts and proc macros are
  themselves binaries.

  ```
  winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

- **macOS:** Xcode command line tools.

Keep the checkout out of any path containing a space. Some build scripts in the
wider Rust and C ecosystem still mishandle them.

## Running

```
npm install
npm run tauri dev
```

Then open settings and paste a Groq API key from
[console.groq.com/keys](https://console.groq.com/keys). The free tier covers
28,800 audio-seconds a day, which is far more than personal dictation uses.

```
cd src-tauri && cargo test    # unit tests, no audio hardware needed
```

## How it fits together

```
mic -> cpal (dedicated thread) -> downmix -> resample 16 kHz -> WAV
                                                                 |
                                                    Groq whisper-large-v3-turbo
                                                                 |
                                                          postprocess()
                                                                 |
                                                            transcript
```

- `audio.rs` - capture, resampling, WAV encoding
- `groq.rs` - transcription client
- `secrets.rs` - API key in the OS credential store
- `settings.rs` - everything else, as JSON in the app config dir
- `lib.rs` - Tauri commands and the `postprocess` seam

### Notes on a few choices

**Audio is captured in Rust, not the webview.** It keeps working when the window
is hidden, which M2's global hotkey depends on, and it sidesteps the WebView2
microphone permission prompt. Capture runs on its own OS thread because
`cpal::Stream` is `!Send` under WASAPI and so can never live in Tauri state.

**The resampler is hand-rolled.** Rubato 5 is built around chunked buffers with
delay bookkeeping, which is more than a one-shot whole-clip convert needs. The
windowed-sinc kernel in `audio.rs` drops its cutoff to the output Nyquist when
downsampling, so content above 8 kHz is attenuated rather than aliasing into the
speech band. Covered by tests.

**`reqwest` is pinned to `native-tls`.** In reqwest 0.13 the `default-tls`
feature means rustls plus aws-lc-rs, which adds a CMake and NASM build step.
`native-tls` uses SChannel on Windows and Security.framework on macOS, so the
build needs no extra tooling.

**`postprocess()` is deliberately near-empty.** The LLM cleanup pass and custom
vocabulary both land there without reshaping the pipeline. It already strips
trailing newlines, because once M2 pastes into the focused window a trailing
newline would submit a terminal prompt the moment the text arrives.

## Cost

Transcription runs on Groq's free tier in normal use. Past that,
`whisper-large-v3-turbo` is $0.04 per hour of audio.

# whispr

Personal dictation app. Tauri, with Groq Whisper in the cloud and local Whisper
as the offline fallback. Windows and macOS.

## Status

Press `Ctrl+Shift+Space` in any app and speak; the transcript is pasted where
you were typing. Tap the key to start and stop, or hold it to talk. Transcription
runs on Groq or on a local Whisper model. Closing the window leaves whispr in the
tray.

- [x] M1 - window app, cpal capture, Groq transcription
- [x] M3 - local Whisper (whisper.cpp), model download, fallback routing
- [x] M2 - global hotkey (tap or hold), auto-paste, tray, recording overlay
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

- **CMake and LLVM**, for whisper.cpp and its bindgen step:

  ```
  winget install -e --id Kitware.CMake
  winget install -e --id LLVM.LLVM
  ```

  If bindgen cannot find libclang, set `LIBCLANG_PATH=C:\Program Files\LLVM\bin`.

- **Optional, NVIDIA GPU:** the CUDA toolkit. Nothing else to do - see below.

- **macOS:** Xcode command line tools.

Keep the checkout out of any path containing a space. Some build scripts in the
wider Rust and C ecosystem still mishandle them.

## Running

```
npm install
npm run tauri dev
```

Then open settings and either download a local model or paste a Groq API key
from [console.groq.com/keys](https://console.groq.com/keys). The Groq free tier
covers 28,800 audio-seconds a day, which is far more than personal dictation uses.

`npm run tauri dev` and `npm run tauri build` pick the engine build for the
machine: with an NVIDIA GPU and the CUDA toolkit they add `--features cuda`,
otherwise whisper.cpp is built for the CPU. The first line of output says which
and why. `WHISPR_GPU=0` or `WHISPR_GPU=1` overrides it. Plain `cargo` commands
skip the detection, so pass `--features cuda` to them yourself.

After installing CUDA, open a **new** terminal. The build finds the toolkit
through `CUDA_PATH_V12_9` (MSBuild's CUDA targets read the versioned variable,
not `CUDA_PATH`), and a shell opened before the install does not have it: CMake
then fails with "The CUDA Toolkit v12.9 directory '' does not exist". The first
CUDA build compiles whisper.cpp's kernels and takes about 9 minutes.

```
cd src-tauri && cargo test    # unit tests, no audio hardware needed

# end to end on the local engine: downloads a model into target/models
WHISPR_TEST_WAV=clip.wav cargo test --lib -- --ignored transcribes_real_speech --nocapture
```

## Installing

```
npm run tauri build
```

This builds a per-user installer (no admin prompt) at
`src-tauri/target/release/bundle/nsis/whispr_<version>_x64-setup.exe`, picking
the GPU or CPU engine the same way `tauri dev` does. Build it on the machine
that will run it: a GPU build needs that machine's CUDA install. The installed
app shares settings, the saved key and downloaded models with the dev build.

In settings, "Start when you log in" adds whispr to the login items; it then
starts straight into the tray with the hotkey ready. The option is greyed out in
dev builds, since the entry would point at a throwaway executable. Launching
whispr while it is already running just opens the running one's window.

## How it fits together

```
mic -> cpal (dedicated thread) -> downmix -> resample 16 kHz
                                                   |
                                  route(): primary engine, then the other
                                   /                              \
                     Groq whisper-large-v3-turbo           whisper.cpp, local
                                   \                              /
                                            postprocess()
                                                   |
                                               transcript
```

- `audio.rs` - capture, resampling, WAV encoding
- `groq.rs` - cloud transcription client
- `local.rs` - local engine, model catalog, checksummed download
- `secrets.rs` - API key in the OS credential store
- `settings.rs` - everything else, as JSON in the app config dir
- `hotkey.rs` - global shortcut, tap-to-toggle vs hold-to-talk
- `paste.rs` - clipboard, the paste keystroke, restoring the old clipboard
- `overlay.rs` - the listening/transcribing pill; never takes focus, click-through
- `lib.rs` - Tauri commands, hotkey sessions, tray, the `postprocess` seam

### Notes on a few choices

**Either engine can be primary.** The other takes over when the primary fails in
a way a retry could fix: offline, rate limited, a server error. A bad key or an
oversized clip does not fall back, because answering from the other engine would
hide a problem the user needs to fix. An engine that is not set up (no key, no
model) is simply skipped.

**Local models are the quantised English-only builds.** Measured on a 6 s clip:

| model | desktop CPU | GTX 1660 SUPER (CUDA) |
|---|---|---|
| `base.en` | 1.2 s, misheard a phrase | - |
| `small.en` | 4.2 s, exact | 0.63 s, exact |
| `large-v3-turbo` | - | 2.0 s, exact |

GPU times are with the model already loaded; the first run after launch adds
about 0.6 s. Small is the default on every build: on the GPU it is three times
faster than Turbo and was just as exact. Every
download is checked against the SHA-256 Hugging Face publishes before it is
renamed into place.

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

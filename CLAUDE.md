# whispr - notes for Claude

Personal dictation app (Tauri 2 + Rust + TypeScript). The README explains the
design; this file is the setup and working checklist.

## Setting up or updating a machine

tomek works on two Windows machines. Work out which one this is first:

```
nvidia-smi -L        # prints "GPU 0: ..." on the desktop, fails on the laptop
```

| | desktop | laptop |
|---|---|---|
| checkout | `D:\dev\whispr` | `C:\dev\whispr` |
| GPU | GTX 1660 SUPER, CUDA 12.9 | none - CPU build |

Keep the checkout out of any path with a space (the laptop user folder is
`C:\Users\Modern 14\`, which breaks C/C++ build scripts).

### Steps

1. **Pull.** `git status` first - if there are local changes, stop and ask
   before touching them. Then `git pull`.

2. **Check the toolchain** and install only what is missing:

   | tool | check | install |
   |---|---|---|
   | Node 20+ | `node -v` | `winget install -e --id OpenJS.NodeJS.LTS` |
   | Rust | `cargo -V` | `winget install -e --id Rustlang.Rustup` |
   | MSVC Build Tools | `vswhere` finds an install | see README prerequisites |
   | CMake | `cmake --version` | `winget install -e --id Kitware.CMake` |
   | LLVM (libclang) | `C:\Program Files\LLVM\bin\libclang.dll` exists | `winget install -e --id LLVM.LLVM` |

   vswhere lives at
   `C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe`.
   Each winget install raises a **UAC prompt** that tomek has to approve - say
   so before starting, since a prompt hidden behind other windows looks like a
   hang (look for `consent.exe` in the process list). Check free space on C:
   first; MSVC needs several GB, and `--installPath D:\...` moves most of it.

3. **Open a new terminal after any install.** Shells started before an install
   do not see its PATH or variables. For CUDA this matters: MSBuild reads
   `CUDA_PATH_V12_9`, and a stale shell fails with "The CUDA Toolkit v12.9
   directory '' does not exist".

4. **Install and run:**

   ```
   npm install
   npm run tauri dev
   ```

   The first output line says which build it picked:
   - laptop: `[whispr] CPU build - no NVIDIA GPU`
   - desktop: `[whispr] GPU (CUDA) build - NVIDIA GPU + CUDA toolkit found`

   The first build compiles whisper.cpp: a few minutes on the CPU build, about
   9 minutes for CUDA. The progress bar sits at ~464/470 the whole time, which
   is normal (`nvcc` or `cl` processes show it working). If bindgen cannot find
   libclang, set `LIBCLANG_PATH=C:\Program Files\LLVM\bin`.

5. **Verify.** In the app: settings, then Local model, then pick Small and
   Download it if needed (190 MB). Then click into Notepad, tap
   `Ctrl+Shift+Space`, speak, tap again. The text should be pasted, and the
   line under the transcript in the whispr window reads `local/small.en (cpu)`
   or `(gpu)`. The live mic test is tomek's - ask him to do it.

6. `cd src-tauri && cargo test --lib` should pass (the ignored tests need a
   microphone or a model download; see README).

## Gotchas

- **Port 1420 in use**: a previous `tauri dev` left Vite running. Find the
  process listening on 1420 and stop it.
- **Testing the hotkey yourself pastes into the focused window** - usually the
  terminal. Room noise is normally rejected by the silence gate, but warn tomek
  and do not rely on it.
- `package-lock.json` shows a diff on machines with a newer npm (it strips
  `libc` fields). It is noise: never commit it.
- whispr stays in the tray when its window is closed; quit from the tray menu.
- The Groq API key is tomek's to create and paste in (settings). Never handle
  it yourself.

## Working rules

- Ask before every commit and push; one approval does not carry to the next.
- No AI attribution anywhere: no Co-Authored-By lines, commits are tomek's.
- Plain hyphens in all text, never em or en dashes.
- Never revert files with `git checkout -- <file>` or `git restore`; undo your
  own edits by editing them back.

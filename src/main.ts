import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { openUrl } from "@tauri-apps/plugin-opener";
import { createFlow } from "./flow";

type Phase = "idle" | "recording" | "transcribing";

type View = "dictate" | "record" | "settings";

const VIEW_TITLES: Record<View, string> = {
  dictate: "Dictate",
  record: "Record",
  settings: "Settings",
};

type Theme = "dark" | "light";

type Engine = "cloud" | "local";

type Language = "en" | "pl";

// Native names, as language pickers usually show them.
const LANGUAGE_NAMES: Record<Language, string> = { en: "English", pl: "Polski" };

interface Settings {
  microphone: string | null;
  engine: Engine;
  model: string;
  local_model: string;
  language: Language;
  hotkey: string;
  auto_paste: boolean;
  cleanup: boolean;
  vocabulary: string[];
  tape_mic: boolean;
  tape_system: boolean;
}

interface TapeStatus {
  recording: boolean;
  elapsed_secs: number;
  path: string | null;
  mic: boolean;
  system: boolean;
}

interface TapeFinished {
  path: string;
  duration_secs: number;
  bytes: number;
}

interface SavedTape {
  name: string;
  path: string;
  bytes: number;
  duration_secs: number;
}

interface Tapes {
  folder: string;
  recordings: SavedTape[];
}

interface Transcript {
  text: string;
  raw_text: string | null;
  cleanup_note: string | null;
  duration_secs: number;
  engine: string;
  language: Language;
  fallback_reason: string | null;
  audio_path: string | null;
}

interface Status {
  recording: boolean;
  has_api_key: boolean;
  has_local_model: boolean;
  // False for an English-only model while dictating in Polish.
  local_speaks: boolean;
  gpu: boolean;
  hotkey_error: string | null;
}

type Delivery = "pasted" | "copied";

// Hotkey sessions run in Rust, even while another app has focus; these events
// let the window follow along.
type DictationEvent =
  | { phase: "recording" }
  | { phase: "transcribing" }
  | { phase: "done"; transcript: Transcript; delivery: Delivery | null }
  | { phase: "error"; message: string };

interface ModelInfo {
  id: string;
  label: string;
  size_bytes: number;
  downloaded: boolean;
  english_only: boolean;
}

interface DownloadProgress {
  id: string;
  received: number;
  total: number;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const els = {
  record: $<HTMLButtonElement>("record"),
  recorder: $<HTMLElement>("recorder"),
  meter: $<HTMLElement>("meter"),
  status: $<HTMLParagraphElement>("status"),
  transcript: $<HTMLTextAreaElement>("transcript"),
  copy: $<HTMLButtonElement>("copy"),
  copyLabel: $<HTMLSpanElement>("copy-label"),
  meta: $<HTMLSpanElement>("meta"),
  error: $<HTMLParagraphElement>("error"),
  settings: $<HTMLElement>("settings"),
  dictate: $<HTMLElement>("view-dictate"),
  navDictate: $<HTMLButtonElement>("nav-dictate"),
  navSettings: $<HTMLButtonElement>("nav-settings"),
  viewTitle: $<HTMLElement>("view-title"),
  engineBadge: $<HTMLElement>("engine-badge"),
  engineLabel: $<HTMLElement>("engine-label"),
  sidebarHotkey: $<HTMLElement>("sidebar-hotkey"),
  apiKey: $<HTMLInputElement>("api-key"),
  saveKey: $<HTMLButtonElement>("save-key"),
  keyStatus: $<HTMLElement>("key-status"),
  groqLink: $<HTMLAnchorElement>("groq-link"),
  mic: $<HTMLSelectElement>("mic"),
  model: $<HTMLSelectElement>("model"),
  engine: $<HTMLSelectElement>("engine"),
  localModel: $<HTMLSelectElement>("local-model"),
  downloadModel: $<HTMLButtonElement>("download-model"),
  downloadProgress: $<HTMLProgressElement>("download-progress"),
  localStatus: $<HTMLElement>("local-status"),
  hotkey: $<HTMLInputElement>("hotkey"),
  saveHotkey: $<HTMLButtonElement>("save-hotkey"),
  hotkeyStatus: $<HTMLElement>("hotkey-status"),
  autoPaste: $<HTMLInputElement>("auto-paste"),
  autostart: $<HTMLInputElement>("autostart"),
  autostartHint: $<HTMLElement>("autostart-hint"),
  cleanup: $<HTMLInputElement>("cleanup"),
  vocabulary: $<HTMLTextAreaElement>("vocabulary"),
  original: $<HTMLDetailsElement>("original"),
  originalText: $<HTMLParagraphElement>("original-text"),
  kbdHint: $<HTMLElement>("kbd-hint"),
  recordView: $<HTMLElement>("view-record"),
  navRecord: $<HTMLButtonElement>("nav-record"),
  navRecordDot: $<HTMLElement>("nav-record-dot"),
  tapeRecorder: $<HTMLElement>("tape-recorder"),
  tapeRecord: $<HTMLButtonElement>("tape-record"),
  tapeStatus: $<HTMLElement>("tape-status"),
  tapeSub: $<HTMLElement>("tape-sub"),
  tapeClock: $<HTMLElement>("tape-clock"),
  tapeError: $<HTMLElement>("tape-error"),
  tapeMic: $<HTMLInputElement>("tape-mic"),
  tapeMicHint: $<HTMLElement>("tape-mic-hint"),
  tapeSystem: $<HTMLInputElement>("tape-system"),
  tapeMicMeter: $<HTMLElement>("tape-mic-meter"),
  tapeSystemMeter: $<HTMLElement>("tape-system-meter"),
  tapes: $<HTMLUListElement>("tapes"),
  tapesEmpty: $<HTMLElement>("tapes-empty"),
  tapesFolder: $<HTMLElement>("tapes-folder"),
  tapesOpen: $<HTMLButtonElement>("tapes-open"),
};

let phase: Phase = "idle";
let settings: Settings = {
  microphone: null,
  engine: "cloud",
  model: "whisper-large-v3-turbo",
  local_model: "small.en",
  language: "en",
  hotkey: "Ctrl+Shift+Space",
  auto_paste: true,
  cleanup: true,
  vocabulary: [],
  tape_mic: true,
  tape_system: true,
};
let models: ModelInfo[] = [];
let downloading = false;
let view: View = "dictate";

const meter = createFlow(els.meter, 36);

const megabytes = (bytes: number) => `${Math.round(bytes / 1_000_000)} MB`;

// --- rendering -------------------------------------------------------------

function setPhase(next: Phase) {
  phase = next;
  els.recorder.dataset.phase = next;
  els.record.setAttribute(
    "aria-label",
    next === "recording" ? "Stop recording" : "Start recording",
  );
  els.record.disabled = next === "transcribing";

  if (next === "recording") {
    els.status.textContent = "Listening...";
    meter.start();
  } else {
    meter.stop();
    if (next === "transcribing") els.status.textContent = "Transcribing...";
  }
}

function showError(message: string) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function clearError() {
  els.error.hidden = true;
  els.error.textContent = "";
}


/** Renders "Ctrl+Shift+Space" as one <kbd> per key. */
function keyChips(combo: string): HTMLElement {
  const wrap = document.createElement("span");
  wrap.className = "keys";
  for (const key of combo.split("+").filter(Boolean)) {
    const kbd = document.createElement("kbd");
    kbd.textContent = key.trim();
    wrap.append(kbd);
  }
  return wrap;
}

function chip(text: string, tone?: "warn" | "note"): HTMLElement {
  const el = document.createElement("span");
  el.className = tone ? `chip ${tone}` : "chip";
  el.textContent = text;
  el.title = text;
  return el;
}

function showView(next: View) {
  view = next;
  els.dictate.hidden = next !== "dictate";
  els.recordView.hidden = next !== "record";
  els.settings.hidden = next !== "settings";
  els.navDictate.classList.toggle("is-active", next === "dictate");
  els.navRecord.classList.toggle("is-active", next === "record");
  els.navSettings.classList.toggle("is-active", next === "settings");
  els.viewTitle.textContent = VIEW_TITLES[next];
  if (next === "record") loadTapes();
}

// --- actions ---------------------------------------------------------------

async function toggleRecording() {
  if (phase === "transcribing") return;
  clearError();

  if (phase === "idle") {
    try {
      await invoke("start_recording");
      setPhase("recording");
    } catch (e) {
      showError(String(e));
      setPhase("idle");
    }
    return;
  }

  setPhase("transcribing");
  try {
    showTranscript(await invoke<Transcript>("stop_and_transcribe"), null);
  } catch (e) {
    showError(String(e));
    els.status.textContent = "Failed";
  } finally {
    setPhase("idle");
  }
}

function showTranscript(result: Transcript, delivery: Delivery | null) {
  els.transcript.value = result.text;
  els.copy.disabled = result.text.length === 0;
  els.meta.replaceChildren(
    chip(`${result.duration_secs.toFixed(1)}s`),
    chip(result.engine),
    chip(LANGUAGE_NAMES[result.language] ?? result.language),
  );
  if (result.fallback_reason) {
    els.meta.append(chip(`Fell back: ${result.fallback_reason}`, "warn"));
  }
  if (result.cleanup_note) els.meta.append(chip(result.cleanup_note, "note"));
  els.original.hidden = result.raw_text === null;
  els.original.open = false;
  els.originalText.textContent = result.raw_text ?? "";
  if (!result.text) {
    els.status.textContent = "Nothing came back";
  } else if (delivery === "pasted") {
    els.status.textContent = "Pasted";
  } else if (delivery === "copied") {
    els.status.textContent = "Copied to the clipboard";
  } else {
    els.status.textContent = "Done";
  }
}

function onDictation(event: DictationEvent) {
  switch (event.phase) {
    case "recording":
      clearError();
      setPhase("recording");
      break;
    case "transcribing":
      setPhase("transcribing");
      break;
    case "done":
      setPhase("idle");
      showTranscript(event.transcript, event.delivery);
      break;
    case "error":
      setPhase("idle");
      showError(event.message);
      els.status.textContent = "Failed";
      break;
  }
}

async function saveHotkey() {
  const combo = els.hotkey.value.trim();
  if (!combo || combo === settings.hotkey) return;
  clearError();
  try {
    await invoke("save_settings", { settings: { ...settings, hotkey: combo } });
    settings = { ...settings, hotkey: combo };
    await refreshStatus();
  } catch (e) {
    // Rejected: show why and put the working hotkey back in the box.
    showError(String(e));
    els.hotkey.value = settings.hotkey;
  }
}

async function loadAutostart() {
  const enabled = await invoke<boolean | null>("get_autostart");
  // null: a dev build, where the login entry would point at a throwaway exe.
  els.autostart.disabled = enabled === null;
  els.autostart.checked = enabled === true;
  if (enabled === null) {
    els.autostartHint.textContent = "Only in the installed app";
  }
}

async function toggleAutostart() {
  clearError();
  try {
    await invoke("set_autostart", { enabled: els.autostart.checked });
  } catch (e) {
    showError(String(e));
    els.autostart.checked = !els.autostart.checked;
  }
}

async function copyTranscript() {
  const text = els.transcript.value;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    els.copyLabel.textContent = "Copied";
    setTimeout(() => (els.copyLabel.textContent = "Copy"), 1200);
  } catch {
    // Fall back to the old selection route if the clipboard API is blocked.
    els.transcript.select();
    document.execCommand("copy");
  }
}

async function refreshStatus() {
  const s = await invoke<Status>("status");
  els.kbdHint.replaceChildren("Press", keyChips("Space"));
  if (!s.hotkey_error) {
    els.kbdHint.append("here, or", keyChips(settings.hotkey), "in any app");
  } else {
    els.kbdHint.append("to start and stop");
  }
  els.sidebarHotkey.replaceChildren(
    ...(s.hotkey_error ? ["Not set"] : keyChips(settings.hotkey).childNodes),
  );
  renderEngineBadge(s);
  els.hotkeyStatus.textContent =
    s.hotkey_error ?? "Works in any app. Tap to start and stop, or hold to talk";
  els.hotkeyStatus.classList.toggle("warn", s.hotkey_error !== null);

  els.keyStatus.textContent = s.has_api_key
    ? "A key is saved in your OS keychain"
    : "No key saved - the cloud engine is off until you add one";
  els.keyStatus.classList.toggle("warn", !s.has_api_key);

  if (!downloading) {
    const model = models.find((m) => m.id === settings.local_model);
    if (!s.has_local_model) {
      els.localStatus.textContent = `Not downloaded - ${megabytes(model?.size_bytes ?? 0)}, one time`;
    } else if (!s.local_speaks) {
      els.localStatus.textContent = `English-only - pick Turbo to dictate in ${
        LANGUAGE_NAMES[settings.language]
      } offline`;
    } else {
      els.localStatus.textContent = `Downloaded - runs on the ${s.gpu ? "GPU" : "CPU"}`;
    }
    els.localStatus.classList.toggle("warn", !s.has_local_model || !s.local_speaks);
    els.downloadModel.hidden = s.has_local_model;
  }

  if (phase === "idle") {
    els.status.textContent =
      s.has_api_key || (s.has_local_model && s.local_speaks)
        ? "Ready"
        : s.has_local_model
          ? "The local model is English-only - pick Turbo or add a Groq key in settings"
          : "Download a local model or add a Groq key in settings";
  }
}

function renderEngineBadge(s: Status) {
  const local = `Local ${settings.local_model} - ${s.gpu ? "GPU" : "CPU"}`;
  const localReady = s.has_local_model && s.local_speaks;
  let label: string;
  let tone: "ok" | "warn" | "error";
  if (settings.engine === "cloud") {
    if (s.has_api_key) [label, tone] = ["Groq cloud", "ok"];
    else if (localReady) [label, tone] = [`${local} (no Groq key)`, "warn"];
    else [label, tone] = ["Not set up", "error"];
  } else {
    if (localReady) [label, tone] = [local, "ok"];
    else if (s.has_api_key && s.has_local_model)
      [label, tone] = ["Groq cloud (local model is English-only)", "warn"];
    else if (s.has_api_key) [label, tone] = ["Groq cloud (model missing)", "warn"];
    else [label, tone] = ["Not set up", "error"];
  }
  els.engineLabel.textContent = label;
  els.engineBadge.dataset.tone = tone;
}

async function saveApiKey() {
  const key = els.apiKey.value.trim();
  if (!key) {
    showError("Paste a key first");
    return;
  }
  clearError();
  try {
    await invoke("save_api_key", { key });
    els.apiKey.value = "";
    await refreshStatus();
  } catch (e) {
    showError(String(e));
  }
}

async function loadMicrophones() {
  const mics = await invoke<string[]>("list_microphones");
  els.mic.innerHTML = "";

  const systemDefault = document.createElement("option");
  systemDefault.value = "";
  systemDefault.textContent = "System default";
  els.mic.append(systemDefault);

  for (const name of mics) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    els.mic.append(opt);
  }
  els.mic.value = settings.microphone ?? "";
}

async function loadLocalModels() {
  models = await invoke<ModelInfo[]>("local_models");
  els.localModel.innerHTML = "";
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    const only = m.english_only ? ", English-only" : "";
    opt.textContent = `${m.label}${only} (${megabytes(m.size_bytes)})`;
    els.localModel.append(opt);
  }
  els.localModel.value = settings.local_model;
}

async function downloadLocalModel() {
  const id = els.localModel.value;
  clearError();
  downloading = true;
  els.downloadModel.disabled = true;
  els.localModel.disabled = true;
  els.downloadProgress.value = 0;
  els.downloadProgress.hidden = false;
  els.localStatus.classList.remove("warn");
  els.localStatus.textContent = "Starting download...";
  try {
    await invoke("download_model", { id });
    await loadLocalModels();
  } catch (e) {
    showError(String(e));
  } finally {
    downloading = false;
    els.downloadModel.disabled = false;
    els.localModel.disabled = false;
    els.downloadProgress.hidden = true;
    await refreshStatus();
  }
}

async function persistSettings() {
  settings = {
    ...settings,
    microphone: els.mic.value || null,
    engine: els.engine.value as Engine,
    model: els.model.value,
    local_model: els.localModel.value,
    auto_paste: els.autoPaste.checked,
    cleanup: els.cleanup.checked,
    vocabulary: els.vocabulary.value.split(/[\n,]/),
    tape_mic: els.tapeMic.checked,
    tape_system: els.tapeSystem.checked,
  };
  try {
    await invoke("save_settings", { settings });
    await refreshStatus();
  } catch (e) {
    showError(String(e));
  }
}

function renderLanguage() {
  for (const b of document.querySelectorAll<HTMLButtonElement>("[data-language]")) {
    const active = b.dataset.language === settings.language;
    b.classList.toggle("is-active", active);
    b.setAttribute("aria-checked", String(active));
  }
}

async function setLanguage(language: Language) {
  if (language === settings.language) return;
  clearError();
  settings = { ...settings, language };
  renderLanguage();
  await persistSettings();
}

// --- meeting recordings ----------------------------------------------------

const tapeMicFlow = createFlow(els.tapeMicMeter, 14, els.tapeMicMeter, "tape_mic_level");
const tapeSystemFlow = createFlow(els.tapeSystemMeter, 14, els.tapeSystemMeter, "tape_system_level");
let tapeRecording = false;
let tapeStartedAt = 0;
let tapeClockTimer: number | undefined;

/** 75 s -> "1:15", 3725 s -> "1:02:05". */
function clock(secs: number): string {
  const s = Math.floor(secs);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

/** Recording sizes: KB under a megabyte, so a short file never reads "0 MB". */
const fileSize = (bytes: number) =>
  bytes < 1_000_000 ? `${Math.max(1, Math.round(bytes / 1000))} KB` : megabytes(bytes);

const fileName = (path: string) => path.split(/[\\/]/).pop() ?? path;

function showTapeError(message: string | null) {
  els.tapeError.textContent = message ?? "";
  els.tapeError.hidden = message === null;
}

/** `saving`: the stop is in flight and the file is being finished. */
function renderTape(status: TapeStatus | null, saving = false) {
  tapeRecording = status?.recording ?? false;
  els.tapeRecorder.dataset.phase = saving ? "transcribing" : tapeRecording ? "recording" : "idle";
  els.tapeRecord.disabled = saving;
  els.tapeRecord.setAttribute("aria-label", tapeRecording ? "Stop recording" : "Start recording");
  els.navRecordDot.hidden = !tapeRecording;
  els.tapeMic.disabled = els.tapeSystem.disabled = tapeRecording || saving;

  if (tapeClockTimer !== undefined) clearInterval(tapeClockTimer);
  tapeClockTimer = undefined;
  tapeMicFlow.stop();
  tapeSystemFlow.stop();
  els.tapeMicMeter.classList.remove("is-live");
  els.tapeSystemMeter.classList.remove("is-live");

  if (saving) {
    els.tapeStatus.textContent = "Saving...";
    return;
  }
  if (!tapeRecording || !status) return;

  tapeStartedAt = performance.now() - status.elapsed_secs * 1000;
  const tick = () => {
    els.tapeClock.textContent = clock((performance.now() - tapeStartedAt) / 1000);
  };
  tick();
  tapeClockTimer = window.setInterval(tick, 250);
  els.tapeStatus.textContent = "Recording";
  els.tapeSub.textContent = status.path ? fileName(status.path) : "";
  if (status.mic) {
    tapeMicFlow.start();
    els.tapeMicMeter.classList.add("is-live");
  }
  if (status.system) {
    tapeSystemFlow.start();
    els.tapeSystemMeter.classList.add("is-live");
  }
}

function renderTapeMicHint() {
  els.tapeMicHint.textContent = `What you say, through ${
    settings.microphone ?? "the default microphone"
  }. Change it on the Dictate page`;
}

async function toggleTape() {
  showTapeError(null);
  if (!tapeRecording) {
    try {
      renderTape(await invoke<TapeStatus>("tape_start"));
    } catch (e) {
      renderTape(null);
      showTapeError(String(e));
    }
    return;
  }
  renderTape(null, true);
  try {
    const done = await invoke<TapeFinished>("tape_stop");
    renderTape(null);
    els.tapeStatus.textContent = "Saved";
    els.tapeSub.textContent = `${fileName(done.path)} - ${fileSize(done.bytes)}`;
    els.tapeClock.textContent = clock(done.duration_secs);
  } catch (e) {
    renderTape(null);
    showTapeError(String(e));
    els.tapeStatus.textContent = "Failed";
  }
  await loadTapes();
}

async function loadTapes() {
  try {
    const t = await invoke<Tapes>("list_tapes");
    els.tapesFolder.textContent = t.folder;
    els.tapesFolder.title = t.folder;
    els.tapesEmpty.hidden = t.recordings.length > 0;
    els.tapes.replaceChildren(
      ...t.recordings.map((r) => {
        const li = document.createElement("li");
        li.className = "tape";
        const name = document.createElement("span");
        name.className = "tape-name";
        name.textContent = r.name.replace(/\.mp3$/i, "");
        name.title = r.path;
        const meta = document.createElement("span");
        meta.className = "tape-meta";
        meta.textContent = `${clock(r.duration_secs)} - ${fileSize(r.bytes)}`;
        const show = document.createElement("button");
        show.className = "btn btn-secondary btn-sm";
        show.type = "button";
        show.textContent = "Show";
        show.addEventListener("click", () =>
          invoke("reveal_tape", { path: r.path }).catch((e) => showTapeError(String(e))),
        );
        li.append(name, meta, show);
        return li;
      }),
    );
  } catch (e) {
    showTapeError(String(e));
  }
}

// --- wiring ----------------------------------------------------------------

els.record.addEventListener("click", toggleRecording);
els.copy.addEventListener("click", copyTranscript);
els.saveKey.addEventListener("click", saveApiKey);
els.mic.addEventListener("change", async () => {
  await persistSettings();
  renderTapeMicHint();
});
els.model.addEventListener("change", persistSettings);
els.engine.addEventListener("change", persistSettings);
els.localModel.addEventListener("change", persistSettings);
els.downloadModel.addEventListener("click", downloadLocalModel);
els.autoPaste.addEventListener("change", persistSettings);
els.autostart.addEventListener("change", toggleAutostart);
els.cleanup.addEventListener("change", persistSettings);
// Saved when the box loses focus, not per keystroke.
els.vocabulary.addEventListener("change", async () => {
  await persistSettings();
  // Show the tidied list (trimmed, de-duplicated) as the backend stored it.
  settings = await invoke<Settings>("get_settings");
  els.vocabulary.value = settings.vocabulary.join("\n");
});
els.saveHotkey.addEventListener("click", saveHotkey);
for (const b of document.querySelectorAll<HTMLButtonElement>("[data-language]")) {
  b.addEventListener("click", () => setLanguage(b.dataset.language as Language));
}
els.hotkey.addEventListener("keydown", (e) => {
  if (e.key === "Enter") saveHotkey();
});

listen<DictationEvent>("dictation", ({ payload }) => onDictation(payload));

listen<DownloadProgress>("model-download", ({ payload }) => {
  els.downloadProgress.value = payload.received / payload.total;
  els.localStatus.textContent = `Downloading... ${megabytes(
    payload.received,
  )} of ${megabytes(payload.total)}`;
});

els.apiKey.addEventListener("keydown", (e) => {
  if (e.key === "Enter") saveApiKey();
});

els.navDictate.addEventListener("click", () => showView("dictate"));
els.navSettings.addEventListener("click", () => showView("settings"));
els.navRecord.addEventListener("click", () => showView("record"));
els.tapeRecord.addEventListener("click", toggleTape);
els.tapeMic.addEventListener("change", persistSettings);
els.tapeSystem.addEventListener("change", persistSettings);
els.tapesOpen.addEventListener("click", () =>
  invoke("open_tapes_folder").catch((e) => showTapeError(String(e))),
);

const appWindow = getCurrentWindow();
$("win-min").addEventListener("click", () => appWindow.minimize());
$("win-max").addEventListener("click", () => appWindow.toggleMaximize());
// Close is intercepted in Rust and hides to the tray.
$("win-close").addEventListener("click", () => appWindow.close());

els.groqLink.addEventListener("click", (e) => {
  e.preventDefault();
  openUrl("https://console.groq.com/keys");
});

// Theme is a per-machine display preference, so localStorage is enough. The
// inline script in index.html applies it before first paint.
const THEME_KEY = "whispr-theme";

function currentTheme(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* storage blocked: the choice only lasts this session */
  }
  for (const b of document.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
    const active = b.dataset.themeChoice === theme;
    b.classList.toggle("is-active", active);
    b.setAttribute("aria-checked", String(active));
  }
  // Keeps the native window border and shadow in step with the page.
  appWindow.setTheme(theme).catch(() => {});
}

for (const b of document.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
  b.addEventListener("click", () => applyTheme(b.dataset.themeChoice as Theme));
}
$("theme-flip").addEventListener("click", () =>
  applyTheme(currentTheme() === "dark" ? "light" : "dark"),
);
applyTheme(currentTheme());

// Ctrl+, opens settings, Escape goes back. Space toggles recording on the
// Dictate page, but not while typing.
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === ",") {
    e.preventDefault();
    showView("settings");
    return;
  }
  if (e.key === "Escape" && view !== "dictate") {
    showView("dictate");
    return;
  }
  if (e.code !== "Space" || e.repeat || view !== "dictate") return;
  const target = e.target as HTMLElement | null;
  const typing =
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement;
  if (typing) return;
  e.preventDefault();
  toggleRecording();
});

async function init() {
  try {
    settings = await invoke<Settings>("get_settings");
    els.model.value = settings.model;
    els.engine.value = settings.engine;
    els.hotkey.value = settings.hotkey;
    els.autoPaste.checked = settings.auto_paste;
    els.cleanup.checked = settings.cleanup;
    renderLanguage();
    els.vocabulary.value = settings.vocabulary.join("\n");
    els.tapeMic.checked = settings.tape_mic;
    els.tapeSystem.checked = settings.tape_system;
    renderTapeMicHint();
    // A recording keeps going with the window closed; pick it back up.
    renderTape(await invoke<TapeStatus>("tape_status"));
    await loadMicrophones();
    await loadLocalModels();
    await loadAutostart();
    await refreshStatus();
    setPhase("idle");
  } catch (e) {
    showError(String(e));
  }
}

init();

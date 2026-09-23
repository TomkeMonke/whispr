import { invoke } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";

type Phase = "idle" | "recording" | "transcribing";

interface Settings {
  microphone: string | null;
  model: string;
  language: string;
}

interface Transcript {
  text: string;
  duration_secs: number;
  engine: string;
  audio_path: string | null;
}

interface Status {
  recording: boolean;
  has_api_key: boolean;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

const els = {
  record: $<HTMLButtonElement>("record"),
  ring: $<HTMLSpanElement>("level-ring"),
  status: $<HTMLParagraphElement>("status"),
  transcript: $<HTMLTextAreaElement>("transcript"),
  copy: $<HTMLButtonElement>("copy"),
  meta: $<HTMLSpanElement>("meta"),
  error: $<HTMLParagraphElement>("error"),
  settings: $<HTMLElement>("settings"),
  settingsToggle: $<HTMLButtonElement>("settings-toggle"),
  apiKey: $<HTMLInputElement>("api-key"),
  saveKey: $<HTMLButtonElement>("save-key"),
  keyStatus: $<HTMLElement>("key-status"),
  groqLink: $<HTMLAnchorElement>("groq-link"),
  mic: $<HTMLSelectElement>("mic"),
  model: $<HTMLSelectElement>("model"),
};

let phase: Phase = "idle";
let settings: Settings = {
  microphone: null,
  model: "whisper-large-v3-turbo",
  language: "en",
};
let levelTimer: number | undefined;

// --- rendering -------------------------------------------------------------

function setPhase(next: Phase) {
  phase = next;
  els.record.classList.toggle("is-recording", next === "recording");
  els.record.disabled = next === "transcribing";

  if (next === "recording") {
    els.status.textContent = "Listening...";
    startLevelPolling();
  } else {
    stopLevelPolling();
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

function startLevelPolling() {
  stopLevelPolling();
  levelTimer = window.setInterval(async () => {
    try {
      const level = await invoke<number>("input_level");
      // Level is a raw peak; a cube root opens up the quiet end so normal
      // speech visibly moves the ring instead of sitting near zero.
      const eased = Math.min(1, Math.cbrt(level) * 1.1);
      els.ring.style.transform = `scale(${1 + eased * 0.55})`;
      els.ring.style.opacity = `${0.25 + eased * 0.65}`;
    } catch {
      /* the stream may already be closing; the next tick will settle it */
    }
  }, 50);
}

function stopLevelPolling() {
  if (levelTimer !== undefined) {
    clearInterval(levelTimer);
    levelTimer = undefined;
  }
  els.ring.style.transform = "scale(1)";
  els.ring.style.opacity = "0";
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
    const result = await invoke<Transcript>("stop_and_transcribe");
    els.transcript.value = result.text;
    els.copy.disabled = result.text.length === 0;
    els.meta.textContent = `${result.duration_secs.toFixed(
      1,
    )}s - ${result.engine}`;
    els.status.textContent = result.text ? "Done" : "Nothing came back";
  } catch (e) {
    showError(String(e));
    els.status.textContent = "Failed";
  } finally {
    setPhase("idle");
  }
}

async function copyTranscript() {
  const text = els.transcript.value;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    els.copy.textContent = "Copied";
    setTimeout(() => (els.copy.textContent = "Copy"), 1200);
  } catch {
    // Fall back to the old selection route if the clipboard API is blocked.
    els.transcript.select();
    document.execCommand("copy");
  }
}

async function refreshKeyStatus() {
  const s = await invoke<Status>("status");
  els.keyStatus.textContent = s.has_api_key
    ? "A key is saved in your OS keychain"
    : "No key saved yet - recording will fail without one";
  els.keyStatus.classList.toggle("warn", !s.has_api_key);
  if (!s.has_api_key && phase === "idle") {
    els.status.textContent = "Add a Groq API key in settings to begin";
  } else if (phase === "idle") {
    els.status.textContent = "Ready";
  }
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
    await refreshKeyStatus();
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

async function persistSettings() {
  settings = {
    ...settings,
    microphone: els.mic.value || null,
    model: els.model.value,
  };
  try {
    await invoke("save_settings", { settings });
  } catch (e) {
    showError(String(e));
  }
}

// --- wiring ----------------------------------------------------------------

els.record.addEventListener("click", toggleRecording);
els.copy.addEventListener("click", copyTranscript);
els.saveKey.addEventListener("click", saveApiKey);
els.mic.addEventListener("change", persistSettings);
els.model.addEventListener("change", persistSettings);

els.apiKey.addEventListener("keydown", (e) => {
  if (e.key === "Enter") saveApiKey();
});

els.settingsToggle.addEventListener("click", () => {
  els.settings.hidden = !els.settings.hidden;
});

els.groqLink.addEventListener("click", (e) => {
  e.preventDefault();
  openUrl("https://console.groq.com/keys");
});

// Space toggles recording, but not while typing.
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
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
    await loadMicrophones();
    await refreshKeyStatus();
    setPhase("idle");
  } catch (e) {
    showError(String(e));
  }
}

init();

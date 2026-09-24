import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// Rust shows and hides this window; the page only reflects the phase.

type Phase = "recording" | "transcribing" | "done" | "error";

type DictationEvent =
  | { phase: "recording" }
  | { phase: "transcribing" }
  | { phase: "done"; transcript: { text: string }; delivery: "pasted" | "copied" | null }
  | { phase: "error"; message: string };

const pill = document.getElementById("pill")!;
const label = document.getElementById("label")!;
const bars = Array.from(document.querySelectorAll<HTMLElement>("#bars i"));

// Different weights per bar so the row moves like a meter, not one block.
const WEIGHTS = [0.55, 0.85, 1, 0.8, 0.5];

let levelTimer: number | undefined;

function setPhase(phase: Phase, text: string) {
  pill.dataset.phase = phase;
  label.textContent = text;
  if (phase === "recording") {
    startLevels();
  } else {
    stopLevels();
  }
}

function startLevels() {
  stopLevels();
  levelTimer = window.setInterval(async () => {
    try {
      const level = await invoke<number>("input_level");
      // Same easing as the main window: a cube root opens up the quiet end.
      const eased = Math.min(1, Math.cbrt(level) * 1.1);
      bars.forEach((bar, i) => {
        const jitter = 0.85 + Math.random() * 0.3;
        const scale = Math.max(0.15, Math.min(1, eased * WEIGHTS[i] * jitter));
        bar.style.transform = `scaleY(${scale})`;
      });
    } catch {
      /* the stream may be closing; the next tick settles it */
    }
  }, 60);
}

function stopLevels() {
  if (levelTimer !== undefined) {
    clearInterval(levelTimer);
    levelTimer = undefined;
  }
  bars.forEach((bar) => (bar.style.transform = "scaleY(0.15)"));
}

listen<DictationEvent>("dictation", ({ payload }) => {
  switch (payload.phase) {
    case "recording":
      setPhase("recording", "Listening");
      break;
    case "transcribing":
      setPhase("transcribing", "Transcribing");
      break;
    case "done":
      if (!payload.transcript.text) {
        setPhase("error", "Nothing heard");
      } else {
        setPhase("done", payload.delivery === "copied" ? "Copied" : "Pasted");
      }
      break;
    case "error":
      setPhase("error", payload.message);
      break;
  }
});

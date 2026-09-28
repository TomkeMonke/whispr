import { listen } from "@tauri-apps/api/event";
import { createFlow } from "./flow";

// Rust shows and hides this window; the page only reflects the phase.

type Phase = "recording" | "transcribing" | "done" | "error";

type DictationEvent =
  | { phase: "recording" }
  | { phase: "transcribing" }
  | { phase: "done"; transcript: { text: string }; delivery: "pasted" | "copied" | null }
  | { phase: "error"; message: string };

const pill = document.getElementById("pill")!;
const label = document.getElementById("label")!;
// The pill's glow follows the voice too, through --flow-level.
const flow = createFlow(document.getElementById("bars")!, 18, pill);

function setPhase(phase: Phase, text: string) {
  pill.dataset.phase = phase;
  label.textContent = text;
  if (phase === "recording") {
    flow.start();
  } else {
    flow.stop();
  }
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

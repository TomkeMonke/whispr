import { invoke } from "@tauri-apps/api/core";

// The live waveform shown while listening, shared by the main window and the
// recording overlay. The mic level is polled, smoothed (fast attack, slow
// release) and drawn every frame as a wave that travels across the bars, with
// a slow breathing pulse underneath so it still looks alive in a pause.

const POLL_MS = 50;
// Level under this counts as room noise (the eased floor sits around 0.15).
const GATE = 0.2;

export interface Flow {
  start(): void;
  stop(): void;
}

/**
 * Fills `container` with `count` bars. The smoothed level is also published as
 * `--flow-level` (0-1) on `levelHost`, for CSS that pulses along with it.
 */
export function createFlow(
  container: HTMLElement,
  count: number,
  levelHost: HTMLElement = container,
): Flow {
  const bars = Array.from({ length: count }, () => {
    const bar = document.createElement("i");
    container.append(bar);
    return bar;
  });
  // Center-weighted envelope, so the middle bars carry most of the motion.
  const envelope = bars.map((_, i) => {
    const x = count === 1 ? 0.5 : i / (count - 1);
    return 0.25 + 0.75 * Math.pow(Math.sin(Math.PI * x), 1.4);
  });

  let target = 0;
  let level = 0;
  let pollTimer: number | undefined;
  let frame: number | undefined;

  const poll = async () => {
    try {
      const raw = await invoke<number>("input_level");
      // A cube root opens up the quiet end, then the gate drops room noise.
      const eased = Math.min(1, Math.cbrt(raw) * 1.1);
      target = Math.max(0, (eased - GATE) / (1 - GATE));
    } catch {
      /* the stream may be closing; the next tick settles it */
    }
  };

  const draw = (now: number) => {
    const t = now / 1000;
    level += (target - level) * (target > level ? 0.35 : 0.08);
    bars.forEach((bar, i) => {
      const x = count === 1 ? 0.5 : i / (count - 1);
      // Two sines at different speeds make the wave flow rather than repeat.
      const wave = 0.6 + 0.4 * Math.sin(t * 7 - x * 10) * Math.sin(t * 2.3 + x * 4);
      const breath = 0.12 + 0.16 * (0.5 + 0.5 * Math.sin(t * 2.4 - x * 5));
      const h = Math.min(1, breath + level * envelope[i] * wave * (1 - breath) * 1.25);
      bar.style.transform = `scaleY(${h.toFixed(3)})`;
      bar.style.opacity = (0.45 + 0.55 * h).toFixed(3);
    });
    levelHost.style.setProperty("--flow-level", level.toFixed(3));
    frame = requestAnimationFrame(draw);
  };

  return {
    start() {
      this.stop();
      target = 0;
      level = 0;
      pollTimer = window.setInterval(poll, POLL_MS);
      frame = requestAnimationFrame(draw);
    },
    stop() {
      if (pollTimer !== undefined) clearInterval(pollTimer);
      if (frame !== undefined) cancelAnimationFrame(frame);
      pollTimer = frame = undefined;
      levelHost.style.setProperty("--flow-level", "0");
      for (const bar of bars) {
        bar.style.transform = "";
        bar.style.opacity = "";
      }
    },
  };
}

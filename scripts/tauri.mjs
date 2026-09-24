// `npm run tauri ...` goes through here so `dev` and `build` pick the GPU build
// on their own: `--features cuda` is added when this machine has an NVIDIA GPU
// and the CUDA toolkit, and left off otherwise (the laptop, or a Mac), where
// whisper.cpp runs on the CPU.
//
// Override the detection with WHISPR_GPU=1 or WHISPR_GPU=0.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

const args = process.argv.slice(2);

function succeeds(command, commandArgs) {
  const result = spawnSync(command, commandArgs, { encoding: "utf8" });
  return result.status === 0 ? result.stdout : null;
}

function hasNvidiaGpu() {
  // nvidia-smi ships with the NVIDIA driver, so it is present exactly when
  // there is an NVIDIA card with a driver installed.
  const out = succeeds("nvidia-smi", ["-L"]);
  return out !== null && /GPU \d+:/.test(out);
}

function hasCudaToolkit() {
  const cudaPath = process.env.CUDA_PATH;
  if (cudaPath && existsSync(cudaPath)) return true;
  return succeeds("nvcc", ["--version"]) !== null;
}

function decideGpu() {
  const forced = process.env.WHISPR_GPU;
  if (forced === "1") return [true, "WHISPR_GPU=1"];
  if (forced === "0") return [false, "WHISPR_GPU=0"];
  if (!hasNvidiaGpu()) return [false, "no NVIDIA GPU"];
  if (!hasCudaToolkit()) {
    return [false, "NVIDIA GPU found but no CUDA toolkit - install it for GPU speed"];
  }
  return [true, "NVIDIA GPU + CUDA toolkit found"];
}

const subcommand = args[0];
const builds = subcommand === "dev" || subcommand === "build";
const featuresGiven = args.some((a) => a === "--features" || a === "-f" || a.startsWith("--features="));

if (builds && !featuresGiven) {
  const [gpu, reason] = decideGpu();
  console.log(`[whispr] ${gpu ? "GPU (CUDA)" : "CPU"} build - ${reason}`);
  if (gpu) args.push("--features", "cuda");
}

// shell: true so Windows resolves node_modules/.bin/tauri.cmd.
const result = spawnSync("tauri", args, { stdio: "inherit", shell: true });
process.exit(result.status ?? 1);

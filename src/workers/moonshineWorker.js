// src/workers/moonshineWorker.js
// 100% Offline Speech-to-Text Worker using Moonshine (onnx-community/moonshine-tiny-ONNX)
// Optimized for edge and high-noise environments with variable-length fast audio decoding.

let pipelinePromise = null;

function postStatus(status) {
  self.postMessage({ type: "STATUS", ...status });
}

async function getTranscriber() {
  if (!pipelinePromise) {
    postStatus({ state: "loading", progress: null });
    pipelinePromise = (async () => {
      const { pipeline, env } = await import("@huggingface/transformers");

      let isCacheAvailable = false;
      try {
        isCacheAvailable = typeof caches !== "undefined" && typeof caches.open === "function";
      } catch {
        isCacheAvailable = false;
      }

      env.useBrowserCache = isCacheAvailable;
      env.useWasmCache = isCacheAvailable;
      env.allowLocalModels = false;
      env.allowRemoteModels = true;

      const modelId = "onnx-community/moonshine-tiny-ONNX";

      const progress_callback = (progress) => {
        if (progress.status === "progress_total" || progress.status === "progress") {
          postStatus({
            state: "loading",
            progress: Math.round(progress.progress ?? 0),
          });
        }
      };

      const transcriber = await pipeline("automatic-speech-recognition", modelId, {
        dtype: "q8",
        device: "wasm",
        progress_callback,
      });

      postStatus({ state: "ready", progress: 100 });
      return transcriber;
    })().catch((err) => {
      pipelinePromise = null;
      postStatus({ state: "error", error: err?.message || String(err) });
      throw err;
    });
  }
  return pipelinePromise;
}

self.onmessage = async ({ data }) => {
  const { id, type, audioData } = data;

  try {
    if (type === "warmup") {
      await getTranscriber();
      self.postMessage({ id, result: { warmed: true } });
      return;
    }

    if (type === "transcribe") {
      const transcriber = await getTranscriber();

      // audioData is a Float32Array of 16kHz mono audio samples
      if (!audioData || audioData.length === 0) {
        self.postMessage({ id, result: { text: "" } });
        return;
      }

      const output = await transcriber(audioData);
      const text = typeof output === "string" ? output : (output?.text || "");
      self.postMessage({ id, result: { text: text.trim() } });
      return;
    }

    self.postMessage({ id, error: `Unknown task type: ${type}` });
  } catch (error) {
    self.postMessage({ id, error: error?.message || "Moonshine transcription failed." });
  }
};

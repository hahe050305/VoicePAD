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

      // Always persist downloaded model files so they are only fetched once.
      // The Cache API is available in Capacitor's WebView (capacitor://localhost),
      // so we unconditionally enable both caches. If the Cache API is somehow
      // unavailable we fall back gracefully — the model will re-download that
      // session but will work correctly.
      let isCacheAvailable = false;
      try {
        isCacheAvailable = typeof caches !== "undefined" && typeof caches.open === "function";
      } catch {
        isCacheAvailable = false;
      }

      env.useBrowserCache = true;        // always try Cache API
      env.useWasmCache = isCacheAvailable;
      env.allowLocalModels = false;
      env.allowRemoteModels = true;

      const modelId = "onnx-community/moonshine-tiny-ONNX";

      const progress_callback = (progress) => {
        if (progress.status === "progress_total") {
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

      // @huggingface/transformers' _call_moonshine defaults max_new_tokens to
      // Math.floor(samples / 16000) * 6, which evaluates to 0 for <1s audio.
      // Ensure a generous token budget so short words and phrases transcribe reliably.
      const durationSec = audioData.length / 16000;
      const max_new_tokens = Math.max(32, Math.ceil(durationSec * 16));

      const output = await transcriber(audioData, { max_new_tokens });
      const rawText = Array.isArray(output)
        ? output[0]?.text || ""
        : typeof output === "string"
          ? output
          : output?.text || "";
      self.postMessage({ id, result: { text: rawText.trim() } });
      return;
    }

    self.postMessage({ id, error: `Unknown task type: ${type}` });
  } catch (error) {
    self.postMessage({ id, error: error?.message || "Moonshine transcription failed." });
  }
};

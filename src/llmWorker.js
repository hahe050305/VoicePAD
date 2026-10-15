// src/llmWorker.js
// Web Worker running a lightweight local LLM in-browser via Transformers.js
import { pipeline, env } from "@huggingface/transformers";

// Configure browser cache and WebGPU support
env.allowLocalModels = false;
env.useBrowserCache = true;

let generator = null;
let isInitializing = false;

const MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct";

async function getGenerator(progressCallback) {
  if (generator) return generator;
  if (isInitializing) {
    while (isInitializing) {
      await new Promise((r) => setTimeout(r, 100));
    }
    return generator;
  }

  isInitializing = true;
  try {
    generator = await pipeline("text-generation", MODEL_ID, {
      dtype: "q4", // 4-bit quantization for minimal size and high speed (~70MB)
      device: "webgpu", // Automatic fallback to WASM if WebGPU is not supported
      progress_callback: progressCallback,
    });
  } catch (err) {
    console.warn("WebGPU initialization failed, falling back to wasm:", err);
    generator = await pipeline("text-generation", MODEL_ID, {
      dtype: "q4",
      device: "wasm",
      progress_callback: progressCallback,
    });
  } finally {
    isInitializing = false;
  }

  return generator;
}

self.onmessage = async (e) => {
  const { type, id, text } = e.data;

  if (type === "INIT") {
    try {
      await getGenerator((progress) => {
        if (progress.status === "progress") {
          self.postMessage({
            type: "INIT_PROGRESS",
            file: progress.file,
            loaded: progress.loaded,
            total: progress.total,
          });
        }
      });
      self.postMessage({ type: "INIT_SUCCESS" });
    } catch (err) {
      self.postMessage({ type: "INIT_ERROR", error: err.message || String(err) });
    }
    return;
  }

  if (type === "PARSE_INTENT") {
    try {
      const model = await getGenerator();

      // Few-shot system prompt tailored for semantic intent & tag mapping
      const messages = [
        {
          role: "system",
          content:
            "Extract search intent from user speech. Output ONLY JSON: {\"target\": \"...\", \"categories\": [\"#...\"]}. Categories can be #vehicle, #security, #document, #health, #bill, #finance, #travel, #parking, #reminder, #contact.",
        },
        {
          role: "user",
          content: "where is my bike parked?",
        },
        {
          role: "assistant",
          content: '{"target": "bike", "categories": ["#vehicle", "#parking"]}',
        },
        {
          role: "user",
          content: "what is the wifi password?",
        },
        {
          role: "assistant",
          content: '{"target": "wifi password", "categories": ["#security"]}',
        },
        {
          role: "user",
          content: text,
        },
      ];

      const output = await model(messages, {
        max_new_tokens: 60,
        temperature: 0.1,
        do_sample: false,
      });

      const responseText = output[0]?.generated_text?.at(-1)?.content || "";

      // Clean and extract JSON object from response
      const jsonMatch = responseText.match(/\{[\s\S]*?\}/);
      let parsed = null;
      if (jsonMatch) {
        try {
          parsed = JSON.parse(jsonMatch[0]);
        } catch {
          parsed = null;
        }
      }

      self.postMessage({
        type: "PARSE_RESULT",
        id,
        result: parsed,
        raw: responseText,
      });
    } catch (err) {
      self.postMessage({
        type: "PARSE_ERROR",
        id,
        error: err.message || String(err),
      });
    }
  }
};

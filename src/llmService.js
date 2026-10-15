// src/llmService.js
// Interface to communicate with the background LLM Web Worker

let worker = null;
let statusListeners = new Set();
let pendingRequests = new Map();
let currentStatus = { state: "idle", progress: 0 }; // idle | loading | ready | error

function getWorker() {
  if (worker) return worker;

  worker = new Worker(new URL("./llmWorker.js", import.meta.url), {
    type: "module",
  });

  worker.onmessage = (e) => {
    const { type, id, result, error, loaded, total } = e.data;

    if (type === "INIT_PROGRESS") {
      const pct = total ? Math.round((loaded / total) * 100) : 0;
      updateStatus({ state: "loading", progress: pct });
    } else if (type === "INIT_SUCCESS") {
      updateStatus({ state: "ready", progress: 100 });
    } else if (type === "INIT_ERROR") {
      updateStatus({ state: "error", error });
    } else if (type === "PARSE_RESULT") {
      const resolver = pendingRequests.get(id);
      if (resolver) {
        resolver.resolve(result);
        pendingRequests.delete(id);
      }
    } else if (type === "PARSE_ERROR") {
      const resolver = pendingRequests.get(id);
      if (resolver) {
        resolver.reject(new Error(error));
        pendingRequests.delete(id);
      }
    }
  };

  return worker;
}

function updateStatus(newStatus) {
  currentStatus = { ...currentStatus, ...newStatus };
  statusListeners.forEach((fn) => fn(currentStatus));
}

export function subscribeLLMStatus(callback) {
  statusListeners.add(callback);
  callback(currentStatus);
  return () => statusListeners.delete(callback);
}

export function initLLM() {
  const w = getWorker();
  updateStatus({ state: "loading", progress: 0 });
  w.postMessage({ type: "INIT" });
}

export function parseIntentWithLLM(text) {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const id = crypto.randomUUID();
    pendingRequests.set(id, { resolve, reject });

    // Set 8-second fallback timeout so the search never hangs
    setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        resolve(null);
      }
    }, 8000);

    w.postMessage({ type: "PARSE_INTENT", id, text });
  });
}

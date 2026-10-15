// src/ocrWorker.js
// Web Worker that runs Tesseract.js OCR on an image Blob.
// NOTE: Tesseract.js will be loaded from CDN (no bundling) to keep bundle size low.

self.importScripts('https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js');

self.onmessage = async (e) => {
  const { imageBlob, id } = e.data;
  try {
    const { data } = await Tesseract.recognize(imageBlob, 'eng', {
      logger: (m) => {
        // optional: post progress back to main thread
        self.postMessage({ id, progress: m.progress });
      },
    });
    self.postMessage({ id, ocr: data.text.trim() });
  } catch (err) {
    self.postMessage({ id, error: err.message || String(err) });
  }
};

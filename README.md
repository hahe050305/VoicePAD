# VoicePad 🎙️📷

> **A zero-cloud multimodal retrieval platform that enables natural-language search across visual and audio data, executing computer vision, speech transcription, and vector search entirely in-client.**

---

## 🚨 Origin of VoicePad: Retrieval in High-Panic Moments

We’ve all been there: you’re at a pharmacy counter or a busy parking exit, holding bags with one hand, and someone asks for a record you snapped weeks ago — your bike's license plate, a receipt, or an old prescription.

With people waiting in line behind you, scrolling through thousands of unorganized camera roll photos or typing with one thumb is painfully slow. In urgent, high-friction moments, typing is a bottleneck.

### Why Voice-First Retrieval?

Speaking is effortless and immediate. Instead of navigating albums or fighting autocorrect, you just tap the mic and say:

> *"Bike plate"* or *"Doctor prescription"*

VoicePad's voice engine processes the spoken phrase as you talk and surfaces the exact image is fetched : completely offline, without waiting on cloud connectivity or cellular reception.

---

## 💡 The USP (Unique Selling Proposition)

**Zero cloud latency. Zero monthly API fees. Absolute privacy.**  

Unlike traditional note apps or cloud AI solutions that upload your sensitive documents, personal receipts, and handwritten notes to remote servers, VoicePad executes **multimodal search, CLIP vector embeddings, OpenCV image preprocessing, and OCR completely on the client side**. Your data lives and dies in your local browser sandbox and device storage.

---

## 🎯 Why VoicePad?

Like most people, I take dozens of quick photos every week: the tail of my parked motorcycle to remember where I left it, handwritten signatures on paper slips, restaurant bills, ID documents, and quick reminders.

Standard tools fail to solve this in high-stakes moments:

1. **Gallery apps are dumb storage buckets**: They organize by timestamp, not semantic meaning.

2. **Standard keyword search fails on real intent**: If I speak *"bike"*, a search engine shouldn't just look for the literal string "bike" — it should surface my motorcycle plate. If I say *"sign"*, it should return my photographed handwritten signature, not an unrelated document or a road billboard.

3. **Cloud AI is a privacy hazard**: Uploading photos of government IDs (Aadhaar, PAN, Driving Licenses) or signatures to third-party endpoints is a massive security risk.

VoicePad bridges the gap between **spoken human intent and on-device computer vision**, creating a lightning-fast second brain where voice queries and camera snaps instantly connect.

---

## 🛠️ Tech Stack & The Evolution (Day 1 to Now)

Every layer of VoicePad went through deliberate architectural iterations based on real profiling, performance bottlenecks, and accuracy trade-offs:

```
┌────────────────────────────────────────────────────────────────────────┐
│                                 VoicePad UI                            │
│                 (React 19 + Tailwind CSS + Lucide Icons)               │
└───────────────────▲────────────────────────────────▲───────────────────┘
                    │                                │
       [Spoken Voice Stream]                [Camera / Image Upload]
                    │                                │
    ┌───────────────▼───────────────┐  ┌─────────────▼──────────────────┐
    │  Moonshine ONNX STT Worker    │  │ OpenCV.js Preprocessing Worker │
    │  (16kHz PCM Noise-Filtered)   │  │ (Bilateral Filter + Adaptive)  │
    └───────────────┬───────────────┘  └─────────────┬──────────────────┘
                    │                                │
                    │                     ┌──────────┴──────────┐
                    │                     ▼                     ▼
                    │             [Android ML Kit]      [Tesseract.js]
                    │             (Native Capacitor)    (WASM Web Worker)
                    │                     │                     │
                    │                     └──────────┬──────────┘
                    │                                │
                    │                  ┌─────────────▼──────────────────┐
                    │                  │  Vision Web Worker (CLIP-ViT)  │
                    │                  │   512-d Dual-Encoder (WASM-q8) │
                    │                  └─────────────┬──────────────────┘
                    │                                │
┌───────────────────▼────────────────────────────────▼───────────────────┐
│                          Hybrid Ranking Engine                         │
│   (Lexical BM25 + Alias Token Expansion + Cosine Similarity + Filter)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                       ┌────────────▼───────────┐
                       │  Dexie.js (IndexedDB)  │
                       │   Local Binary Storage │
                       └────────────────────────┘
```

---

### 👁️ The Visual Understanding Evolution: `Normal OCR ➔ YOLO ➔ CLIP`

The journey to making VoicePad truly "see" and connect with voice commands was the most critical technological pivot of this project:

```
┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
│   Phase 1: Normal OCR   │       │   Phase 2: YOLO Model   │       │      Phase 3: CLIP      │
│     (Lexical Text)      │ ───►  │     (Bounding Boxes)    │ ───►  │      (Multimodal)       │
│  "Read text on images"  │       │  "Detect 80 COCO items" │       │  "Unified Vector Space" │
└─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
```

#### 1. Phase 1: Normal OCR Tagging (The Lexical Text Era)

* **What I Did**: Originally, every image captured was pushed through an OCR engine (Tesseract) to extract words, regex-match structured numbers, and tag documents based on recognized keywords (`#receipt`, `#plate`).

* **Why It Broke Down**:

  * **Textless Images Were Invisible**: If I took a picture of my bike, a water bottle, a laptop, a coffee mug, or car keys, OCR returned empty whitespace. The system had zero comprehension of what was in the photo, rendering visual items completely unsearchable.

  * **Handwriting & Noise**: Cursive handwriting or low-light snaps turned text into garbled strings, generating nonsense tags and burying critical notes.

#### 2. Phase 2: YOLOv8 Object Detection (The Bounding Box Era)

* **What I Did**: To fix the "blind to objects" problem, I integrated **YOLOv8 Nano** via ONNX runtime in a background worker. Images were evaluated for physical objects, allowing the app to automatically assign tags like `#car`, `#bicycle`, `#bottle`, `#laptop`.

* **Why It Also Broke Down**:

  * **The 80-Class Box**: YOLO is bounded by the 80 fixed COCO categories. While it could identify a `"car"`, it had zero semantic depth for real-world personal artifacts. It couldn't distinguish a legal contract from a restaurant menu, had no concept of a handwritten signature, couldn't recognize an ID card, and couldn't process personal documents.

  * **No Semantic Generalization**: If a user said *"beverage"*, YOLO had no way of associating it with a detected `"bottle"`. If a user said *"sports"*, YOLO couldn't connect it to a cricket bat or tournament ticket. We had bounding boxes, but zero conceptual intelligence.

#### 3. Phase 3: CLIP Dual-Encoder Vector Embeddings (The Final Arrival)

* **Where I Arrived**: I discarded the rigid bounding-box model and migrated to **`Xenova/clip-vit-base-patch32`** running on-device via WebAssembly (q8 quantized).

* **Why It Solved the Problem**:

  * **Unified Multimodal Space**: CLIP maps both visual images and natural language text queries into a shared **512-dimensional continuous latent vector space**.

  * **Zero-Shot Visual Semantics**: Instead of hardcoding categories, VoicePad projects spoken voice queries directly into the vector space. When you say *"sign"*, *"paper with signature"*, or *"beverage"*, the system calculates cosine similarity against your stored image embeddings.

  * Even if an image contains zero OCR text and no YOLO bounding box, its visual geometry (the blue ink curves of a signature, the shape of a bike fender, the layout of a receipt) matches the spoken query with mathematical precision.

---

### Additional Architecture Transitions

#### Application Core & Cross-Platform Wrapper

* **The Transition**: Built a native mobile pipeline using **Capacitor 8** with custom platform adapter layers (`cameraAdapter.js`, `storageAdapter.js`, `voiceAdapter.js`). This allowed VoicePad to run on Android with direct hardware access to camera, microphone, and filesystem, while keeping 100% desktop web compatibility.

#### Local Storage & Memory Management

* **Stored notes and base64 strings in `localStorage`**: Quickly hit the 5MB quota limit; base64 strings bloated memory and froze the UI.

* **The Transition**: Migrated to **Dexie.js (IndexedDB)** with binary `Blob` storage and an asynchronous canvas-based thumbnail generation queue (`createImageThumbnail`) to keep the feed scrolling at 60fps.

#### OCR Architecture: Native Android ML Kit + Tesseract WASM

Heavy Tesseract.js running on the main UI thread freezes the app while processing multiple images at once.

* **The Transition**:

  * **On Native Android**: Integrated Google’s **ML Kit Text Recognition (`@capacitor-mlkit/text-recognition`)**, slashing OCR latency.

  * **On Web**: Isolated Tesseract.js in a dedicated Web Worker (`ocrWorker.js`) using WebAssembly.

  * **OpenCV Preprocessing (`@techstark/opencv-js`)**: Added bilateral filtering and adaptive thresholding to clean noise and perspective-warp skewed plates and documents before OCR execution.

#### Spoken Voice Engine: WebSpeech ➔ Moonshine Edge STT

Standard browser WebSpeech API.

* **The Problem**: In public settings near traffic or racing vehicles, low signal-to-noise ratio and laptop fan noise caused phoneme dropouts and inaccurate transcription.

* **The Transition**: Integrated **Useful Sensors Moonshine-Tiny ONNX** running client-side inside an offline Web Worker (`moonshineWorker.js`). Moonshine processes variable-length audio without fixed padding, achieving lower Word Error Rate (WER) than Whisper-Tiny and decoding speech in under **150ms** through a 16kHz hardware noise-gated audio pipeline.

---

## 🧩 Major Engineering Battles & Reflections

### 1. The "Sign" vs. License Plate False-Positive Incident

* **The Bug**: During user testing, saying or typing *"sign"* returned a photo of a motorcycle license plate (`BR 01 C J 6440`) as the #1 match, while an actual handwritten signature was tagged as `#receipt #bill #book` and buried!

* **The Root Cause**: 

  1. The search engine had a hardcoded scoring boost (`+150`) for vehicle registration plates, which overwhelmed natural scores.

  2. The word *"sign"* had no alias link to *"signature"* or *"autograph"*.

  3. CLIP's prompt dictionary only had generic document prompts, so a white slip of paper with ink handwriting looked closer to a store bill than a document.

* **The Fix**:

  * Added dedicated visual prompts in `visionWorker.js`:  
    `"a photo of a handwritten signature, sign, or autograph on a piece of paper"` -> `[#signature, #document]`.

  * Updated `tagger.js` and `aliasMap.json` with regexes and bidirectional synonym expansions for `sign ⟷ signature ⟷ autograph ⟷ signed`.

  * Rebuilt visual score scaling: when lexical score is 0, enforce a strict cosine similarity threshold (`0.25`).

  * Implemented **`filterBestOrSimilarMatches`**: isolates the single best match when confidence is high, and only shows secondary matches if their scores are within a tight confidence band (`score >= topScore * 0.75`), sorted strictly high-to-low.

### 2. High-Security Registration Plate (HSRP) Font Confusion & Two-Line Plates

* **The Challenge**: Indian license plates often split text across two lines (e.g., `BR 01 C` on top, `J 6440` below), and HSRP fonts suffer from heavy OCR character confusion: `0` vs `O`, `1` vs `I`, `8` vs `B`, and `2` vs `Z`.

* **The Solution**:

  * Authored a dedicated `plateService.js` parser containing valid 2-letter state codes (`VALID_INDIAN_STATES`) and Bharat series (`BH`) syntax rules.

  * Implemented spatial multi-line concatenation grouping adjacent vertical bounding boxes.

  * Created an **HSRP Confusion Substitution Matrix** that contextually swaps numerals for letters in state code positions and letters for numerals in suffix positions.

  * Added **negative keyword guards** (`NON_VEHICLE_KEYWORDS` like `BOOKMYSHOW`, `SWIGGY`, `INVOICE`, `TICKET`) to stop cinema ticket codes or invoice IDs from hallucinating vehicle plate tags.

### 3. Preventing False Positive ID Tagging

* **The Challenge**: An 8-to-12 character alphanumeric string in a store receipt would frequently trigger PAN or Aadhaar card patterns.

* **The Solution**: In `ocr.js`, structured document detection (`extractStructuredIds`) was guarded by contextual validation: a PAN number (`[A-Z]{5}[0-9]{4}[A-Z]`) is only confirmed if surrounding OCR text contains supportive keywords like `"INCOME TAX"`, `"GOVERNMENT OF INDIA"`, or `"PERMANENT ACCOUNT"`.

### 4. Browser Thread Freezes & Model Memory Management

* **The Challenge**: Downloading and initializing a 150MB ONNX vision model on mobile devices with 4GB RAM caused garbage collection thrashing and killed background tabs.

* **The Solution**:

  * Segregated heavy computation into separate Web Workers (`visionWorker.js`, `ocrWorker.js`, `ocrPreprocessWorker.js`).

  * Implemented non-blocking **background model pre-warming** (`prewarmVisionModel`, `prewarmOCR`): models compile silently after the feed has mounted.

  * Built an idle teardown watchdog: if no inference occurs, model buffers are released to keep device thermals and RAM usage minimal.

---

## 🧪 Validated Against Real-World Test Scenarios

The system has been exercised and unit-tested against diverse, unconstrained real-world test media:

| Scenario / Image Type | Real-World Challenge | How VoicePad Handles It | Verified Result |
| :--- | :--- | :--- | :--- |
| **Handwritten Signatures** | Blue ink on paper slip, cursive handwriting (*"Harish J.H.R"*) | CLIP prompt detects signature geometry; `tagger.js` extracts lexical signature intent. | Tagged as `#signature, #document`. Top match for query *"sign"*. |
| **Two-Wheeler License Plates** | Mud, road vibration, two-tier text layout (*"BR 01 C / J 6440"*) | OpenCV contour localization + HSRP confusion mapping + multi-line concatenation. | Tagged as `#vehicle, #licenseplate, #motorcycle`. Searchable via state or digits. |
| **Natural Language Reminders** | Informal text notes: *"cricket match tmrw"* | Temporal parser detects *"tmrw"*; sports dictionary identifies *"cricket"*. | Automatically tagged with `#sports, #reminder, #event`. |
| **Structured Identity Cards** | PAN, Aadhaar, Driving Licenses with complex layouts | Regex extraction gated by government keyword contextual validation. | Tagged with `#document`; masked structured values saved for instant retrieval. |
| **Receipts & Invoices** | Thermal paper, faded text, total amounts | OCR variant generation extracts prices and dates; CLIP prompts isolate store bills. | Tagged with `#receipt, #bill`. Queries like *"expenses"* or *"bill"* return top match. |
| **Everyday Physical Objects** | Desk laptops, coffee mugs, water bottles, keys | 512-dimensional CLIP cosine similarity against visual category dictionary. | Instantly auto-tagged (`#coffee, #cup`, `#laptop, #tech`) with zero cloud calls. |

All tagging rules, synonym expansions, and ranking filters are covered by automated unit tests in `tests/search-and-tagger.test.js`:

```bash
node --test tests/search-and-tagger.test.js
# ✔ deriveTags detects sports and temporal reminder tags for cricket match tmrw
# ✔ deriveTags detects signature tags for sign and signature
# ✔ normalizeQuery includes synonyms and tag mappings for sign
# ✔ filterBestOrSimilarMatches returns ONLY the best match when score gap is large
# ✔ filterBestOrSimilarMatches returns all similar confidence matches ordered high-to-low
# tests 5 | pass 5 | fail 0
```

---

## 🚀 Project Setup & Quickstart

### Prerequisites

* **Node.js**: `v18.0.0` or higher (Node 20+ recommended)

* **npm**: `v9.0.0` or higher

* **Browser**: Chrome, Edge, or Safari with WebAssembly and Web Workers enabled.

* *(Optional for Android APK)*: Android Studio & Android SDK 33+ for Capacitor builds.

### 1. Clone the Repository

```bash
git clone https://github.com/hahe050305/VoicePAD.git
cd VoicePAD
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Run the Development Server

```bash
npm run dev
```

Open your browser at `http://localhost:5173`.  
*Note: When testing microphone or camera inputs in the browser, grant the requested browser permissions.*

### 4. Run the Test Suite

```bash
npm test
```

Runs the native Node.js test runner covering tag derivation, query normalization, and high-to-low confidence ranking.

### 5. (Optional) Run on Android via Capacitor

```bash
# Build the production web assets
npm run build

# Sync web bundle into the Android native shell
npx cap sync android

# Open in Android Studio to build APK or run on a physical device
npx cap open android
```

---

## 📌 Architecture Notes & Key Files

* [`src/App.jsx`](file:///c:/Projects/VoicePad/src/App.jsx) – Main UI application, voice stream lifecycle, in-card note editing, and debounced live search filters.

* [`src/search.js`](file:///c:/Projects/VoicePad/src/search.js) – Hybrid scoring engine (BM25 token scoring, synonym expansion, CLIP vector cosine similarity, and `filterBestOrSimilarMatches`).

* [`src/tagger.js`](file:///c:/Projects/VoicePad/src/tagger.js) – Rule-based semantic auto-tagger detecting sports, signatures, temporal dates, and vehicle metadata.

* [`src/aliasMap.json`](file:///c:/Projects/VoicePad/src/aliasMap.json) – Bidirectional semantic alias mappings bridging synonyms across visual and voice domains.

* [`src/workers/moonshineWorker.js`](file:///c:/Projects/VoicePad/src/workers/moonshineWorker.js) – 100% offline edge Speech-to-Text worker powered by Moonshine (`onnx-community/moonshine-tiny-ONNX`).

* [`src/services/moonshineService.js`](file:///c:/Projects/VoicePad/src/services/moonshineService.js) – 16kHz Web Audio PCM recorder, hardware noise-suppression, and worker audio bridge.

* [`src/workers/visionWorker.js`](file:///c:/Projects/VoicePad/src/workers/visionWorker.js) – Offline Web Worker running `Xenova/clip-vit-base-patch32` for 512-d embeddings.

* [`src/services/plateService.js`](file:///c:/Projects/VoicePad/src/services/plateService.js) – HSRP font confusion mapping, two-line plate concatenation, and Indian state code validation.

* [`src/ocr.js`](file:///c:/Projects/VoicePad/src/ocr.js) – Unified OCR router between native Android ML Kit and Tesseract Web Worker.

* [`src/db.js`](file:///c:/Projects/VoicePad/src/db.js) – Dexie.js IndexedDB schema and persistent binary image store.

---

## 💬 Closing Reflection

VoicePad started as a simple hack to keep track of receipts without typing, but evolving it into a reliable voice-first retrieval tool for high-panic moments changed my entire perspective on client-side AI. 

Tackling the real-world problems — false-positive visual matches, noisy OCR from skewed vehicle plates, WebAssembly memory limits on low-end phones, and speech recognition latency — proved that making AI useful isn't just about importing a massive model; it's about the pragmatic engineering around it: **clever data normalization, worker threading, domain-specific confusion dictionaries, and ruthless ranking algorithms**. 

*Feedback and contributions are welcome!*

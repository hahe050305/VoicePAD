// src/App.jsx
import { useEffect, useState, useRef, useCallback } from "react";
import { db, seedDemoData } from "./db.js";
import { rankedSearch, fastSearch, refreshCache } from "./search.js";
import { performOCR, isPureImageText } from "./ocr.js";
import { deriveTags } from "./tagger.js";
import { initLLM, subscribeLLMStatus } from "./llmService.js";
import { getDetectedLabelsAndTags} from "./services/visionService.js";
import { createId } from "./id.js";

export default function App() {
  const [query, setQuery] = useState("");
  const [voiceTranscript, setVoiceTranscript] = useState("");
  const [searchMode, setSearchMode] = useState("all"); // "voice" | "text" | "all"
  const [filteredEntries, setFilteredEntries] = useState([]);
  const [allEntries, setAllEntries] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [manualNote, setManualNote] = useState("");
  const [autoTags, setAutoTags] = useState([]);
  const [llmStatus, setLlmStatus] = useState({ state: "idle", progress: 0 });

  // In-card editing state
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState("");
  const [editLabel, setEditLabel] = useState("");
  const [editingError, setEditingError] = useState("");

  // Zoom modal state — entry selected by clicking a card
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [ocrExpandedId, setOcrExpandedId] = useState(null);

  // Keep the feed at the same viewport position while a card is zoomed.
  useEffect(() => {
    if (!selectedEntry) return undefined;

    const scrollY = window.scrollY;
    const root = document.documentElement;
    const body = document.body;
    const previousRootOverflow = root.style.overflow;
    const previousBodyOverflow = body.style.overflow;
    const previousBodyPaddingRight = body.style.paddingRight;
    const scrollbarWidth = window.innerWidth - root.clientWidth;

    root.style.overflow = "hidden";
    body.style.overflow = "hidden";
    if (scrollbarWidth > 0) {
      body.style.paddingRight = `${scrollbarWidth}px`;
    }
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setSelectedEntry(null);
    };
    window.addEventListener("keydown", closeOnEscape);

    return () => {
      window.removeEventListener("keydown", closeOnEscape);
      root.style.overflow = previousRootOverflow;
      body.style.overflow = previousBodyOverflow;
      body.style.paddingRight = previousBodyPaddingRight;
      if (window.scrollY !== scrollY) {
        window.scrollTo(0, scrollY);
      }
    };
  }, [selectedEntry]);

  // Top-right label reminder popup for newly uploaded unlabeled images
  const [showLabelReminder, setShowLabelReminder] = useState(false);
  const reminderTimerRef = useRef(null);

  const recognizerRef = useRef(null);
  const fileInputRef = useRef(null);
  const searchDebounceRef = useRef(null);
  const imageProcessingQueueRef = useRef(Promise.resolve());

  //const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  // Load initial entries
  const loadEntries = useCallback(async () => {
    await seedDemoData();
    const entries = await refreshCache();
    const sorted = [...entries].sort((a, b) => b.timestamp - a.timestamp);
    setAllEntries(sorted);
    setFilteredEntries(sorted.slice(0, 20));
  }, []);

  useEffect(() => {
    void Promise.resolve()
      .then(loadEntries)
      .catch((err) => console.error("Failed to load memories:", err));
    const unsub = subscribeLLMStatus(setLlmStatus);
    // Initialize LLM in the background
    initLLM();

   // prewarmOCR();
    //prewarmDetector();
  //  prewarmDeepLearningOCR();
    return () => {
      unsub();
      if (reminderTimerRef.current) clearTimeout(reminderTimerRef.current);
    };
  }, [loadEntries]);

  // ── Voice recognition ────────────────────────────────────────────────────────
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognizer = new SpeechRecognition();
    recognizer.continuous = false;
    recognizer.interimResults = true;
    recognizer.lang = "en-US";

    recognizer.onstart = () => {
      setIsListening(true);
    };

    recognizer.onend = () => {
      setIsListening(false);
    };

    recognizer.onerror = (e) => {
      console.debug("Speech recognition error:", e.error);
      setIsListening(false);
    };

    recognizer.onresult = (e) => {
      // 1. Immediately extract latest spoken words (interim + final)
      let currentTranscript = "";
      for (let i = 0; i < e.results.length; i++) {
        currentTranscript += e.results[i][0].transcript + " ";
      }
      currentTranscript = currentTranscript.trim();

      if (!currentTranscript) return;

      // 2. Zero-latency instant visual reflection in the Instagram Notes floater bubble
      setVoiceTranscript(currentTranscript);
      setSearchMode("voice");

      // 3. Debounce background search so fast ongoing speaking does not thrash, but search begins instantly
      if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
      setIsSearching(true);

      searchDebounceRef.current = setTimeout(async () => {
        try {
          const result = await fastSearch(currentTranscript, (instantHit) => {
            if (instantHit) {
              const hits = Array.isArray(instantHit) ? instantHit : [instantHit];
              setFilteredEntries(hits);
              setIsSearching(false);
            }
          });

          if (result) {
            const hits = Array.isArray(result) ? result : [result];
            setFilteredEntries(hits);
          } else {
            setFilteredEntries([]);
          }
        } finally {
          setIsSearching(false);
        }
      }, 60);
    };

    recognizerRef.current = recognizer;
  }, []);

  const toggleListening = () => {
    if (!recognizerRef.current) {
      alert("Voice recognition not supported in this browser.");
      return;
    }

    if (isListening) {
      setIsListening(false);
      try {
        recognizerRef.current.stop();
      } catch {
        // Ignore if already stopped
      }
    } else {
      // Immediate optimistic visual feedback (<1ms)
      setIsListening(true);
      setVoiceTranscript("");
      try {
        recognizerRef.current.start();
      } catch {
        // If already starting or in an invalid state, abort and restart cleanly
        try {
          recognizerRef.current.abort();
          recognizerRef.current.start();
        } catch {
          setIsListening(false);
        }
      }
    }
  };

  const clearVoiceSearch = () => {
    setVoiceTranscript("");
    setSearchMode("all");
    setFilteredEntries(allEntries.slice(0, 20));
  };

  // ── Live feed filtering for manual text search (debounced) ───────────────────
  const handleLiveFilter = useCallback(
    (q) => {
      if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
      if (!q || !q.trim()) {
        setIsSearching(false);
        setSearchMode("all");
        setFilteredEntries(allEntries.slice(0, 20));
        return;
      }
      setSearchMode("text");
      setIsSearching(true);
      searchDebounceRef.current = setTimeout(async () => {
        try {
          const ranked = await rankedSearch(q);
          setFilteredEntries(ranked.slice(0, 20));
        } finally {
          setIsSearching(false);
        }
      }, 150);
    },
    [allEntries]
  );

  const handleQueryChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    handleLiveFilter(val);
  };

  const handleSearchSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    setSearchMode("text");
    setIsSearching(true);
    try {
      const ranked = await rankedSearch(query);
      setFilteredEntries(ranked.slice(0, 20));
    } finally {
      setIsSearching(false);
    }
  };

  // ── Quick note with semantic auto-tagging ────────────────────────────────────
  const handleNoteChange = (e) => {
    const val = e.target.value;
    setManualNote(val);
    setAutoTags(deriveTags(val));
  };

  const handleAddQuickNote = async (e) => {
    e.preventDefault();
    if (!manualNote.trim()) return;

    const tags = deriveTags(manualNote);
    const newEntry = {
      id: createId(),
      type: "text",
      content: manualNote.trim(),
      tags,
      timestamp: Date.now(),
    };

    await db.notes.add(newEntry);
    setManualNote("");
    setAutoTags([]);
    const updated = await refreshCache();
    const sorted = [...updated].sort((a, b) => b.timestamp - a.timestamp);
    setAllEntries(sorted);
    if (query.trim()) {
      const ranked = await rankedSearch(query);
      setFilteredEntries(ranked.slice(0, 20));
    } else {
      setFilteredEntries(sorted.slice(0, 20));
    }
  };

  // ── In-card editing ──────────────────────────────────────────────────────────
  const startEditing = (entry, e) => {
    if (e) e.stopPropagation();
    setEditingId(entry.id);
    setEditText(entry.type === "text" ? entry.content || "" : entry.ocr || "");
    setEditLabel(entry.label || "");
    setEditingError("");
  };

  const cancelEditing = (e) => {
    if (e) e.stopPropagation();
    setEditingId(null);
    setEditText("");
    setEditLabel("");
    setEditingError("");
  };

  const handleSaveEdit = async (entry, e) => {
    if (e) e.stopPropagation();
    const updatedText = editText.trim();
    const updatedLabel = editLabel.trim();

    if (!updatedText && !updatedLabel) {
      setEditingError("Text or label cannot be empty.");
      return;
    }
    setEditingError("");

    if (entry.type === "text") {
      const updatedTags = deriveTags(updatedText);
      await db.notes.update(entry.id, {
        content: updatedText,
        tags: updatedTags,
      });
    } else {
      const autoTags = extractTagsFromText(updatedText);
      let labelTags = [];
      if (updatedLabel) {
        const normalized = updatedLabel.toLowerCase().replace(/[^a-z0-9]/g, "_");
        const derived = deriveTags(updatedLabel);
        labelTags = [`#label_${normalized}`, `#${normalized}`, ...derived];
      }
      const combinedTags = Array.from(new Set([...labelTags, ...autoTags]));

      await db.images.update(entry.id, {
        ocr: updatedText,
        label: updatedLabel,
        tags: combinedTags,
      });
    }

    // Refresh cache and lists
    const updated = await refreshCache();
    const sorted = [...updated].sort((a, b) => b.timestamp - a.timestamp);
    setAllEntries(sorted);

    if (query.trim()) {
      const ranked = await rankedSearch(query);
      setFilteredEntries(ranked.slice(0, 20));
    } else {
      setFilteredEntries(sorted.slice(0, 20));
    }

    setEditingId(null);
    setEditText("");
    setEditLabel("");
  };

  // ── Image Capture / OCR ──────────────────────────────────────────────────────
  const extractTagsFromText = (text) => {
    if (!text || !text.trim()) return ["#image", "#photo"];
    const tags = deriveTags(text);
    return tags.length > 0 ? Array.from(new Set([...tags, "#image"])) : ["#image", "#photo"];
  };

  const handleImageCapture = async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    // Clear immediately so the same selection can be added again while this
    // batch is being indexed in the background.
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";

    try {
      // Store the selected files as Blobs; this avoids creating and retaining
      // base64 copies of every selected image in memory.
      const newEntries = files.map((file, i) => ({
        id: createId(),
        type: "image",
        imageBlob: file,
        label: "",
        ocr: "",
        isPureImage: true, // Default to true so it renders instantly as a photo placeholder
        tags: ["#image", "#photo"],
        timestamp: Date.now() + i,
      }));

      // 3. Immediately store to database and update UI
      await db.images.bulkAdd(newEntries);
      const updated = await refreshCache();
      const sorted = [...updated].sort((a, b) => b.timestamp - a.timestamp);
      setAllEntries(sorted);
      setFilteredEntries(sorted.slice(0, 20));

      // 4. Trigger top-right reminder popup on every unlabelled image after a few seconds
      if (reminderTimerRef.current) clearTimeout(reminderTimerRef.current);
      reminderTimerRef.current = setTimeout(() => {
        setShowLabelReminder(true);
      }, 2500);

      // 4. Run OCR pipeline SILENTLY in the background without blocking or showing progress
      const processingTask = imageProcessingQueueRef.current.then(async () => {
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          const entry = newEntries[i];
          try {
            console.log(`[Vision] Processing image ${i + 1}/${files.length}...`);
            // Run OCR first, then detect objects.
            const ocrResult = await performOCR(file).catch((err) => {
              console.warn("[Vision] OCR error:", err);
              return "";
            });

            const extractedText = typeof ocrResult === "string" ? ocrResult : (ocrResult?.text || "");
            const activePlate = typeof ocrResult === "object" && ocrResult?.plate ? ocrResult.plate : null;
            const structuredIds = typeof ocrResult === "object" && Array.isArray(ocrResult?.structuredIds) ? ocrResult.structuredIds : [];
            const cleanText = extractedText.trim();
            const validWords = cleanText
              .replace(/[^a-zA-Z\s]/g, " ")
              .split(/\s+/)
              .filter((word) => word.length >= 3);

            const hasVehiclePlate = !!activePlate;
            const hasStructuredId = structuredIds.length > 0;
            const isTextDominant = hasVehiclePlate || hasStructuredId || validWords.length >= 2;

            let visionResult = { labels: [], tags: [], detections: [] };
            if (!isTextDominant) {
              visionResult = await getDetectedLabelsAndTags(file).catch((err) => {
                console.warn("[Vision] Vision detection error:", err);
                return { labels: [], tags: [], detections: [] };
              });
            }

            const isValidTag = (tag) => {
              const clean = tag.replace("#", "").toLowerCase();
              if (clean.length < 3 || !/[aeiou0-9]/.test(clean) || /(.)\1\1/.test(clean)) return false;
              return !["npn", "wwn", "broic", "ji", "er", "te"].includes(clean);
            };

            const textTags = cleanText ? deriveTags(cleanText) : [];
            const headlineWordTags = validWords.slice(0, 4).map((word) => `#${word.toLowerCase()}`);
            const activeVisionTags = isTextDominant ? [] : (visionResult.tags || []);
            const idTags = structuredIds.map((id) => `#${id.value.toLowerCase()}`);
            const rawCombinedTags = [
              ...textTags,
              ...headlineWordTags,
              ...activeVisionTags,
              ...idTags,
              ...(hasVehiclePlate ? ["#vehicle", `#${activePlate.toLowerCase()}`] : []),
              ...(hasStructuredId ? ["#document"] : []),
              ...(isTextDominant && !hasVehiclePlate && !hasStructuredId ? ["#poster", "#graphic"] : ["#photo"]),
            ];
            const combinedTags = Array.from(new Set(rawCombinedTags.filter(isValidTag)));
            const finalOcrText = activePlate && !cleanText.includes(activePlate)
              ? `${activePlate}\n${cleanText}`.trim()
              : cleanText;

            await db.images.update(entry.id, {
              ocr: finalOcrText,
              isPureImage: !hasVehiclePlate && !hasStructuredId && !isTextDominant && finalOcrText.length === 0,
              detectedObjects: isTextDominant ? [] : visionResult.labels,
              tags: combinedTags,
            });
          } catch (err) {
            console.error("[Vision] Silent background processing failed:", err);
          }
        }
        // Re-sync UI state so the new tags immediately appear on the cards!
        const refreshed = await refreshCache();
        const refreshedSorted = [...refreshed].sort((a, b) => b.timestamp - a.timestamp);
        setAllEntries(refreshedSorted);
        if (query.trim()) {
          const ranked = await rankedSearch(query);
          setFilteredEntries(ranked.slice(0, 20));
        } else {
          setFilteredEntries(refreshedSorted.slice(0, 20));
        }
      });
      imageProcessingQueueRef.current = processingTask.catch((err) => {
        console.error("[Vision] Background image processing failed:", err);
      });
    } catch (err) {
      console.error("Real‑timeimage upload failed:", err);
      alert("Error adding images. Please try again.");
    }
  };

  // ── Delete ───────────────────────────────────────────────────────────────────
  const handleDeleteEntry = async (entry, e) => {
    if (e) e.stopPropagation();
    // if (!window.confirm("Delete this memory?")) return;

    if (entry.type === "image") await db.images.delete(entry.id);
    else await db.notes.delete(entry.id);

    const updated = await refreshCache();
    const sorted = [...updated].sort((a, b) => b.timestamp - a.timestamp);
    setAllEntries(sorted);
    if (query.trim()) {
      const ranked = await rankedSearch(query);
      setFilteredEntries(ranked.slice(0, 20));
    } else {
      setFilteredEntries(sorted.slice(0, 20));
    }
  };

  const formatTime = (ts) =>
    new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center px-4 py-8 antialiased selection:bg-indigo-500 selection:text-white">
      {/* Top-Right Notification: Appears on every unlabelled image upload after a few seconds */}
      {showLabelReminder && (
        <div className="fixed top-5 right-5 z-50 max-w-sm animate-in fade-in slide-in-from-top-4 duration-300">
          <div className="bg-slate-900/95 border border-amber-500/50 shadow-2xl rounded-2xl p-3.5 backdrop-blur-md flex items-start gap-3 text-slate-100">
            <span className="text-xl shrink-0 mt-0.5">🏷️</span>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-amber-300">
                Unlabeled image uploaded
              </p>
              <p className="text-[11px] text-slate-300 mt-0.5 leading-snug">
                Add a quick label (e.g. "Selfie", "License") so voice search retrieves it with 100% precision!
              </p>
              <div className="flex items-center gap-2 mt-2">
                <button
                  onClick={() => {
                    const firstUnlabeled = allEntries.find((e) => e.type === "image" && !e.label);
                    if (firstUnlabeled) {
                      startEditing(firstUnlabeled);
                    }
                    setShowLabelReminder(false);
                  }}
                  className="px-2.5 py-1 bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-lg text-[11px] font-semibold transition shadow"
                >
                  Label Now
                </button>
                <button
                  onClick={() => setShowLabelReminder(false)}
                  className="px-2 py-1 text-[11px] text-slate-400 hover:text-slate-200 transition"
                >
                  Dismiss
                </button>
              </div>
            </div>
            <button
              onClick={() => setShowLabelReminder(false)}
              className="text-slate-500 hover:text-slate-300 text-xs p-1"
              aria-label="Close reminder"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="w-full max-w-xl flex items-center justify-between pb-6 border-b border-slate-800/80 mb-8">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse"></span>
            VoicePAD
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Real‑time
          </p>
        </div>
                <div className="flex items-center gap-2">
          {llmStatus.state === "loading" && !llmStatus.isCached && (
            <span className="text-[11px] font-mono tracking-wider px-2 py-0.5 rounded-full bg-indigo-950/80 border border-indigo-700/50 text-indigo-300 animate-pulse">
              🧠 Downloading AI ({llmStatus.progress}%)
            </span>
          )}
          {llmStatus.state === "cached_loading" && (
            <span className="text-[11px] font-mono tracking-wider px-2 py-0.5 rounded-full bg-indigo-950/50 border border-indigo-800/40 text-indigo-300/80">
              ⚡ Initializing...
            </span>
          )}
          {llmStatus.state === "ready" && (
            <span className="text-[11px] font-mono tracking-wider px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-700/50 text-emerald-300">
              ⚡ Local AI Ready
            </span>
          )}
          <span className="text-[11px] font-mono tracking-wider uppercase px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-slate-400">
            Local-First
          </span>
        </div>
      </header>

      <main className="w-full max-w-xl flex flex-col items-center">
        {/* Voice Orb with Instagram Notes-inspired Floater Bubble */}
        <div className="relative my-6 flex flex-col items-center">
          {/* Instagram Notes Floater Bubble (Top-Right of Mic) */}
          {(voiceTranscript || isListening) && (
            <div className="absolute -top-10 -right-8 z-20 flex flex-col items-start max-w-[210px] animate-in fade-in zoom-in-95 duration-200">
              <div className="relative px-3 py-1.5 bg-indigo-600 text-white rounded-2xl rounded-bl-sm shadow-xl border border-indigo-400/40 flex items-center gap-1.5 backdrop-blur-md">
                <span className="text-xs">💭</span>
                <span className="text-xs font-medium tracking-tight truncate max-w-[150px]">
                  {voiceTranscript ? `"${voiceTranscript}"` : "Listening..."}
                </span>
                {voiceTranscript && (
                  <button
                    onClick={clearVoiceSearch}
                    className="ml-1 text-[11px] text-indigo-200 hover:text-white px-1 py-0.5 rounded-full hover:bg-indigo-700/60 transition"
                    title="Clear voice query"
                    aria-label="Clear voice query"
                  >
                    ✕
                  </button>
                )}
              </div>
              {/* Instagram Notes speech bubble pointer dot */}
              <div className="w-2 h-2 bg-indigo-600 rounded-full ml-2 -mt-0.5 border border-indigo-400/30" />
            </div>
          )}

          {isListening && (
            <div className="absolute inset-0 -m-2 rounded-full bg-rose-500/25 blur-md animate-pulse pointer-events-none" />
          )}
          <button
            onClick={toggleListening}
            className={`relative z-10 w-28 h-28 rounded-full flex flex-col items-center justify-center transition-transform duration-100 shadow-2xl active:scale-95 select-none ${
              isListening
                ? "bg-rose-600 shadow-rose-600/40 ring-4 ring-rose-500/40 scale-105"
                : "bg-indigo-600 hover:bg-indigo-500 shadow-indigo-600/30 ring-4 ring-indigo-500/20"
            }`}
            aria-label="Activate voice search"
          >
            <span className="text-3xl mb-1 select-none">{isListening ? "🛑" : "🎙️"}</span>
            <span className="text-[10px] font-semibold tracking-wider uppercase text-white/90 select-none">
              {isListening ? "Listening" : "Tap to Ask"}
            </span>
          </button>
        </div>

        {/* Image Capture — gallery supports selecting multiple images */}
        <div className="w-full flex items-center justify-center gap-3 my-2">
          <input
    ref={fileInputRef}
    type="file"
    accept="image/*"
    multiple
    className="hidden"
    onChange={handleImageCapture}
  />
  <input
    ref={cameraInputRef}
    type="file"
    accept="image/*"
    capture="environment"
    className="hidden"
    onChange={handleImageCapture}
  />
          <button
    type="button"
    onClick={() => cameraInputRef.current?.click()}
    className="flex-1 max-w-[170px] flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-xl text-xs font-semibold border transition shadow-sm bg-indigo-600 hover:bg-indigo-500 border-indigo-500/40 text-white active:scale-95"
  >
    <span className="text-base">📸</span>
    <span>Live Capture</span>
  </button>
  {/* 2. Gallery / File Upload */}
  <button
    type="button"
    onClick={() => fileInputRef.current?.click()}
    className="flex-1 max-w-[170px] flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-xl text-xs font-medium border transition shadow-sm bg-slate-900 hover:bg-slate-800 border-slate-800 hover:border-slate-700 text-slate-200 active:scale-95"
  >
    <span className="text-base">📁</span>
    <span>Upload Photos</span>
  </button>
        </div>

        {/* Search Bar — manual text filter */}
        <form onSubmit={handleSearchSubmit} className="w-full mt-4">
          <div className="relative flex items-center">
            <span className="absolute left-3 text-slate-500 text-sm pointer-events-none">
              🔍
            </span>
            <input
              type="text"
              placeholder="Search memories… filters feed as you type"
              value={query}
              onChange={handleQueryChange}
              className="w-full pl-9 pr-24 py-3 bg-slate-900/90 border border-slate-800 rounded-xl text-sm placeholder-slate-500 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition shadow-inner"
            />
            {query && (
              <button
                type="button"
                onClick={() => {
                  setQuery("");
                  setSearchMode("all");
                  setFilteredEntries(allEntries.slice(0, 20));
                }}
                className="absolute right-20 text-slate-500 hover:text-slate-300 text-xs"
              >
                ✕
              </button>
            )}
            
          </div>
          {/* Show active filter indicator for manual text */}
          {query.trim() && searchMode === "text" && (
            <p className="text-[11px] text-slate-500 mt-1.5 ml-1">
              Showing{" "}
              <span className="text-indigo-400 font-medium">
                {filteredEntries.length}
              </span>{" "}
              match{filteredEntries.length !== 1 ? "es" : ""} for "{query}"
            </p>
          )}
        </form>

        {/* Quick Note with auto-tagging */}
        <form onSubmit={handleAddQuickNote} className="w-full mt-3">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Jot a note — tags auto-detected from what you type…"
              value={manualNote}
              onChange={handleNoteChange}
              className="flex-1 px-3 py-2 bg-slate-900/50 border border-slate-800/80 rounded-lg text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:border-slate-700"
            />
            <button
              type="submit"
              className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium border border-slate-700/80 transition whitespace-nowrap"
            >
              + Save
            </button>
          </div>
          {/* Live tag preview */}
          {manualNote.trim() && autoTags.length > 0 && (
            <div className="flex items-center gap-1.5 mt-2 flex-wrap">
              <span className="text-[10px] text-slate-500">Auto-tags:</span>
              {autoTags.map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-950/70 text-indigo-300 border border-indigo-800/50 font-medium"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </form>

        {/* Feed — with in-card editing */}
        <section className="w-full mt-8">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              {searchMode === "voice" && voiceTranscript.trim() ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                  {filteredEntries.length > 1 ? `Your Photos (${filteredEntries.length})` : "Top Voice Match"}
                </>
              ) : query.trim() ? (
                "Best Matches"
              ) : (
                `Stored Memories (${allEntries.length})`
              )}
            </h2>
            <span className="text-[11px] text-slate-500">
              {searchMode === "voice" && voiceTranscript.trim() ? (
                <button
                  onClick={clearVoiceSearch}
                  className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2 transition"
                >
                  Show all memories
                </button>
              ) : query.trim() ? (
                `${filteredEntries.length} results`
              ) : (
                "Auto-Indexed"
              )}
            </span>
          </div>

          {isSearching ? (
            <div className="space-y-2.5 animate-pulse" aria-label="Loading search results">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="p-3.5 bg-slate-900/40 border border-slate-800/60 rounded-xl flex items-start justify-between gap-3"
                >
                  <div className="flex-1 space-y-2">
                    <div className="h-3.5 bg-slate-800 rounded w-3/4" />
                    <div className="h-3 bg-slate-800/70 rounded w-1/2" />
                    <div className="flex gap-1.5 pt-1">
                      <div className="h-4 bg-slate-800/80 rounded w-12" />
                      <div className="h-4 bg-slate-800/80 rounded w-14" />
                    </div>
                  </div>
                  <div className="w-10 h-3 bg-slate-800/50 rounded shrink-0" />
                </div>
              ))}
            </div>
          ) : filteredEntries.length === 0 && (query.trim() || voiceTranscript.trim()) ? (
            <div className="py-10 text-center">
              <p className="text-slate-400 text-sm">
                No matching memory found for "{voiceTranscript || query}"
              </p>
              <p className="text-slate-600 text-xs mt-1">
                Try asking differently or check your stored notes below.
              </p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {filteredEntries.map((entry) => {
                const isEditing = editingId === entry.id;
                const isPureImage =
                  entry.type === "image" &&
                  (entry.isPureImage === true || (!entry.ocr && !entry.label) || isPureImageText(entry.ocr));

                return (
                  <div
                    key={entry.id}
                    onClick={() => {
                      if (!isEditing) {
                        setOcrExpandedId(null);
                        setSelectedEntry(entry);
                      }
                    }}
                    className="p-3.5 bg-slate-900/70 border border-slate-800/80 hover:border-slate-700 rounded-xl transition flex flex-col gap-2.5 shadow-sm cursor-pointer"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                          {entry.type === "image" && !isPureImage ? (
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation();
                                setOcrExpandedId((currentId) => currentId === entry.id ? null : entry.id);
                              }}
                              aria-expanded={ocrExpandedId === entry.id}
                              aria-label={`${ocrExpandedId === entry.id ? "Hide" : "Show"} extracted OCR text`}
                              title={`${ocrExpandedId === entry.id ? "Hide" : "Show"} extracted text`}
                              className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
                            >
                              📷 OCR {ocrExpandedId === entry.id ? "⌃" : "⌄"}
                            </button>
                          ) : (
                            <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                              {entry.type === "image" ? "🖼️ Image" : "📝 Note"}
                            </span>
                          )}
                          {entry.label ? (
                            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-indigo-950/90 border border-indigo-500/40 text-indigo-300 flex items-center gap-1">
                              🏷️ {entry.label}
                            </span>
                          ) : entry.type === "image" ? (
                            <button
                              onClick={(e) => startEditing(entry, e)}
                              className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-amber-950/80 border border-amber-600/40 text-amber-300 hover:bg-amber-900/90 transition flex items-center gap-1"
                              title="Click to add a label for instant search"
                            >
                              🏷️ + Label
                            </button>
                          ) : null}
                        </div>

                        {/* Inline Edit Form OR Card Content */}
                        {isEditing ? (
                          <div className="mt-1 space-y-2">
                            {entry.type === "image" && (
                              <div>
                                <label className="text-[10px] text-slate-400 block mb-1 font-medium">
                                  Image Label:
                                </label>
                                <input
                                  type="text"
                                  value={editLabel}
                                  onChange={(e) => setEditLabel(e.target.value)}
                                  className="w-full px-2.5 py-1.5 bg-slate-950 border border-indigo-500/50 rounded-lg text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 mb-2"
                                  placeholder="e.g. My Photo, Aadhaar Card, License Plate..."
                                  autoFocus
                                />
                              </div>
                            )}
                            <textarea
                              value={editText}
                              onChange={(e) => setEditText(e.target.value)}
                              rows={isPureImage ? 2 : 3}
                              className="w-full p-2.5 bg-slate-950 border border-indigo-500/50 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                              placeholder={
                                isPureImage
                                  ? "Optional description / caption..."
                                  : entry.type === "image"
                                  ? "Extracted OCR text..."
                                  : "Edit memory content..."
                              }
                            />
                            {/* Live preview of auto-tags while editing */}
                            {(editText.trim() || editLabel.trim()) && (
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <span className="text-[10px] text-slate-500">
                                  Updated tags:
                                </span>
                                {deriveTags(`${editLabel} ${editText}`).map((tag) => (
                                  <span
                                    key={tag}
                                    className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-950/70 text-indigo-300 border border-indigo-800/50 font-medium"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            )}
                            <div className="flex items-center gap-2 pt-1">
                              <button
                                onClick={(e) => handleSaveEdit(entry, e)}
                                className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition shadow"
                              >
                                Save
                              </button>
                              <button
                                onClick={cancelEditing}
                                className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium transition"
                              >
                                Cancel
                              </button>
                            </div>
                            {editingError && (
                              <p className="text-[11px] text-rose-400 mt-1 font-medium animate-in fade-in">
                                ⚠️ {editingError}
                              </p>
                            )}
                          </div>
                        ) : entry.type === "image" ? (
                          <div className="mt-1 space-y-2">
                            <div className="w-full max-h-80 rounded-xl overflow-hidden bg-slate-950 border border-slate-800/90 flex items-center justify-center p-1">
                              <img
                                src={entry.blobUrl}
                                alt={entry.label || "Image"}
                                className="max-h-72 max-w-full object-contain rounded-lg"
                              />
                            </div>
                            {!isPureImage && ocrExpandedId === entry.id && entry.ocr && (
                              <p
                                id={`ocr-text-${entry.id}`}
                                className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap"
                              >
                                {entry.ocr}
                              </p>
                            )}
                            <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                              {(entry.tags?.some((t) => !["#image", "#photo", "#picture", "#scooty", "#scooter", "#twowheeler", "#fourwheeler", "#automobile"].includes(t))
                                ? entry.tags.filter((t) => !["#image", "#photo", "#picture", "#scooty", "#scooter", "#twowheeler", "#fourwheeler", "#automobile"].includes(t))
                                : entry.tags?.filter((t) => t !== "#image")
                              )?.map((tag) => (
                                <span
                                  key={tag}
                                  className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-950/70 text-indigo-300 border border-indigo-800/50 font-medium"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : (
                          /* Standard Text or OCR Document */
                          <>
                            <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                              {entry.type === "text"
                                ? entry.content
                                : entry.ocr || "(No readable text in image)"}
                            </p>
                            <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                              {(entry.tags?.some((t) => !["#image", "#photo", "#picture", "#scooty", "#scooter", "#twowheeler", "#fourwheeler", "#automobile"].includes(t))
                                ? entry.tags.filter((t) => !["#image", "#photo", "#picture", "#scooty", "#scooter", "#twowheeler", "#fourwheeler", "#automobile"].includes(t))
                                : entry.tags?.filter((t) => t !== "#image")
                              )?.map((tag) => (
                                <span
                                  key={tag}
                                  className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-950/70 text-indigo-300 border border-indigo-800/50 font-medium"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </>
                        )}
                      </div>

                      {/* Right Action Icons (Edit, Delete, Timestamp) */}
                      {!isEditing && (
                        <div className="flex flex-col items-end gap-1.5 shrink-0">
                          <span className="text-[10px] text-slate-500 font-mono">
                            {formatTime(entry.timestamp)}
                          </span>
                          <div className="flex items-center gap-1">
                            <button
                              onClick={(e) => startEditing(entry, e)}
                              className="text-[12px] text-slate-400 hover:text-indigo-400 p-1 rounded hover:bg-slate-800 transition"
                              title="Edit memory"
                              aria-label="Edit memory"
                            >
                              ✏️
                            </button>
                            <button
                              onClick={(e) => handleDeleteEntry(entry, e)}
                              className="text-[12px] text-slate-400 hover:text-rose-400 p-1 rounded hover:bg-slate-800 transition"
                              title="Delete memory"
                              aria-label="Delete memory"
                            >
                              🗑️
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </main>

      {/* ── Centered zoomed memory card ───────────────────────────────────── */}
      {selectedEntry && (() => {
        const modalEntry = selectedEntry;
        const modalPure =
          modalEntry.type === "image" &&
          (modalEntry.isPureImage === true || (!modalEntry.ocr && !modalEntry.label) || isPureImageText(modalEntry.ocr));
        const modalTags = modalEntry.tags?.some((tag) =>
          !["#image", "#photo", "#picture", "#scooty", "#scooter", "#twowheeler", "#fourwheeler", "#automobile"].includes(tag)
        )
          ? modalEntry.tags.filter((tag) =>
              !["#image", "#photo", "#picture", "#scooty", "#scooter", "#twowheeler", "#fourwheeler", "#automobile"].includes(tag)
            )
          : modalEntry.tags?.filter((tag) => tag !== "#image");

        return (
          <div
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-150"
            onClick={() => setSelectedEntry(null)}
          >
            <div
              className="w-full max-w-3xl max-h-[90vh] overflow-y-auto p-3.5 bg-slate-900/70 border border-slate-800/80 rounded-xl shadow-2xl animate-in zoom-in-95 duration-150"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                  {modalEntry.type === "image" && !modalPure ? (
                    <button
                      type="button"
                      onClick={() => setOcrExpandedId((currentId) => currentId === modalEntry.id ? null : modalEntry.id)}
                      aria-expanded={ocrExpandedId === modalEntry.id}
                      aria-label={`${ocrExpandedId === modalEntry.id ? "Hide" : "Show"} extracted OCR text`}
                      className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
                    >
                      📷 OCR {ocrExpandedId === modalEntry.id ? "⌃" : "⌄"}
                    </button>
                  ) : (
                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                      {modalEntry.type === "image" ? "🖼️ Image" : "📝 Note"}
                    </span>
                  )}
                  {modalEntry.label ? (
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-indigo-950/90 border border-indigo-500/40 text-indigo-300 flex items-center gap-1">
                      🏷️ {modalEntry.label}
                    </span>
                  ) : modalEntry.type === "image" ? (
                    <button
                      onClick={() => { startEditing(modalEntry); setSelectedEntry(null); }}
                      className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-amber-950/80 border border-amber-600/40 text-amber-300 hover:bg-amber-900/90 transition flex items-center gap-1"
                    >
                      🏷️ + Label
                    </button>
                  ) : null}
                </div>
                <div className="flex flex-col items-end gap-1.5 shrink-0">
                  <span className="text-[10px] text-slate-500 font-mono">
                    {formatTime(modalEntry.timestamp)}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => { startEditing(modalEntry); setSelectedEntry(null); }}
                      className="text-[12px] text-slate-400 hover:text-indigo-400 p-1 rounded hover:bg-slate-800 transition"
                      title="Edit memory"
                      aria-label="Edit memory"
                    >
                      ✏️
                    </button>
                    <button
                      onClick={(event) => { handleDeleteEntry(modalEntry, event); setSelectedEntry(null); }}
                      className="text-[12px] text-slate-400 hover:text-rose-400 p-1 rounded hover:bg-slate-800 transition"
                      title="Delete memory"
                      aria-label="Delete memory"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              </div>

              {modalEntry.type === "image" ? (
                <div className="mt-1 space-y-2">
                  <div className="w-full max-h-[70vh] rounded-xl overflow-hidden bg-slate-950 border border-slate-800/90 flex items-center justify-center p-1 group">
                    <img
                      src={modalEntry.blobUrl}
                      alt={modalEntry.label || "Image"}
                      className="max-h-[68vh] w-full object-contain rounded-lg"
                    />
                  </div>
                  {!modalPure && ocrExpandedId === modalEntry.id && modalEntry.ocr && (
                    <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                      {modalEntry.ocr}
                    </p>
                  )}
                </div>
              ) : (
                <div className="mt-1">
                  <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                    {modalEntry.content}
                  </p>
                </div>
              )}

              {modalTags?.length > 0 && (
                <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                  {modalTags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-950/70 text-indigo-300 border border-indigo-800/50 font-medium"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
}

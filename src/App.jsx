// src/App.jsx
import { useEffect, useState, useRef, useCallback } from "react";
import { db, seedDemoData } from "./db.js";
import { rankedSearch, fastSearch, lexicalSearch, refreshCache, getImageThumbnailUrl, setImageThumbnailUrl, getImageSource } from "./search.js";
import { performOCR, recognizeVehiclePlate, isPureImageText } from "./ocr.js";
import { deriveTags } from "./tagger.js";
import {
  createImageThumbnail,
  embedImage,
  subscribeVisionModelStatus,
  prewarmVisionModel,
} from "./services/visionService.js";
import { prewarmOCR } from "./services/tesseractService.js";
import {
  prewarmMoonshine,
  subscribeMoonshineStatus,
  startAudioCapture,
  stopAudioCaptureAndTranscribe,
  cancelAudioCapture,
} from "./services/moonshineService.js";
import { createId } from "./id.js";

export default function App() {
  const [query, setQuery] = useState("");
  const [voiceTranscript, setVoiceTranscript] = useState("");
  const [searchMode, setSearchMode] = useState("all"); // "voice" | "text" | "all"
  const [filteredEntries, setFilteredEntries] = useState([]);
  const [allEntries, setAllEntries] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [moonshineStatus, setMoonshineStatus] = useState({
    state: "idle",
    progress: null,
    error: null,
  });
  const [visionModelStatus, setVisionModelStatus] = useState({
    state: "idle",
    progress: null,
    error: null,
  });
  const [manualNote, setManualNote] = useState("");
  const [autoTags, setAutoTags] = useState([]);

  // In-card editing state
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState("");
  const [editLabel, setEditLabel] = useState("");
  const [editingError, setEditingError] = useState("");

  // Card-level Voice Note & Quick Manual Labeling Fallback state
  const [recordingCardId, setRecordingCardId] = useState(null);
  const cardRecognizerRef = useRef(null);
  const [quickLabelingId, setQuickLabelingId] = useState(null);
  const [quickLabelInput, setQuickLabelInput] = useState("");

  // Zoom modal state — entry selected by clicking a card
  const [selectedEntry, setSelectedEntry] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
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

  useEffect(() => {
    if (selectedEntry?.type !== "image") return undefined;

    let cancelled = false;
    let imageUrl;
    getImageSource(selectedEntry.id)
      .then((url) => {
        imageUrl = url;
        if (cancelled) {
          if (url.startsWith("blob:")) URL.revokeObjectURL(url);
        } else {
          setSelectedImage({ id: selectedEntry.id, url });
        }
      })
      .catch((error) => {
        console.error("[Image] Could not load full-size image:", error);
      });

    return () => {
      cancelled = true;
      if (imageUrl?.startsWith("blob:")) URL.revokeObjectURL(imageUrl);
    };
  }, [selectedEntry?.id, selectedEntry?.type]);

  const recognizerRef = useRef(null);
  const audioStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioAnalyserRef = useRef(null);
  const fileInputRef = useRef(null);
  const searchDebounceRef = useRef(null);
  const imageProcessingQueueRef = useRef(Promise.resolve());
  const thumbnailJobsRef = useRef(new Set());

  const cameraInputRef = useRef(null);

  const enqueueThumbnailBackfill = useCallback((entries) => {
    const candidates = entries.filter((entry) =>
      entry.type === "image" &&
      !entry.thumbnailUrl &&
      !entry.thumbnailFailed &&
      !thumbnailJobsRef.current.has(entry.id)
    );
    if (candidates.length === 0) return;
    candidates.forEach((entry) => thumbnailJobsRef.current.add(entry.id));

    const task = imageProcessingQueueRef.current.then(async () => {
      for (const entry of candidates) {
        try {
          const record = await db.images.get(entry.id);
          const source = record?.imageBlob || record?.blobUrl;
          if (!source) throw new Error("Image source is unavailable.");
          const thumbnail = await createImageThumbnail(source);
          if (thumbnail) {
            await db.images.update(entry.id, { thumbnailBlob: thumbnail });
          } else {
            await db.images.update(entry.id, { thumbnailFailed: true });
          }
          const thumbnailUrl = thumbnail ? getImageThumbnailUrl(entry.id, thumbnail) : null;
          const updatePreview = (image) => image.id === entry.id
            ? { ...image, thumbnailUrl, thumbnailFailed: !thumbnail }
            : image;
          setFilteredEntries((current) => current.map(updatePreview));
          setSelectedEntry((current) => current?.id === entry.id ? updatePreview(current) : current);
        } catch (error) {
          console.warn("[Vision] Could not prepare saved image preview:", error);
          await db.images.update(entry.id, { thumbnailFailed: true });
          setFilteredEntries((current) => current.map((image) =>
            image.id === entry.id ? { ...image, thumbnailFailed: true } : image
          ));
        } finally {
          thumbnailJobsRef.current.delete(entry.id);
        }
      }
      const refreshed = await refreshCache();
      setAllEntries([...refreshed].sort((a, b) => b.timestamp - a.timestamp));
    });
    imageProcessingQueueRef.current = task.catch((error) => {
      console.error("[Vision] Saved image preview backfill failed:", error);
      candidates.forEach((entry) => thumbnailJobsRef.current.delete(entry.id));
    });
  }, []);

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
    const unsubscribeVisionStatus = subscribeVisionModelStatus(setVisionModelStatus);

    // Prior prewarming: silently warms up Vision AI, OCR, and Moonshine STT in background
    prewarmVisionModel();
    prewarmOCR();
    prewarmMoonshine();

    const unsubscribeMoonshine = subscribeMoonshineStatus(setMoonshineStatus);

    return () => {
      unsubscribeVisionStatus();
      unsubscribeMoonshine();
    };
  }, [loadEntries]);

  useEffect(() => {
    enqueueThumbnailBackfill(filteredEntries);
  }, [filteredEntries, enqueueThumbnailBackfill]);

  // ── Laptop Microphone Noise-Suppression & Audio Pre-Filtering ──────────────
  const stopAudioPipeline = useCallback(() => {
    if (audioStreamRef.current) {
      try {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
      } catch (err) {
        console.debug("[AudioPipeline] Error stopping stream tracks:", err);
      }
      audioStreamRef.current = null;
    }
    if (audioContextRef.current) {
      try {
        if (audioContextRef.current.state !== "closed") {
          audioContextRef.current.close().catch(() => {});
        }
      } catch (err) {
        console.debug("[AudioPipeline] Error closing AudioContext:", err);
      }
      audioContextRef.current = null;
    }
    audioAnalyserRef.current = null;
  }, []);

  const startAudioPipeline = useCallback(async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return null;
    }
    try {
      // Hardware-level constraints: Acoustic Echo Cancellation, Noise Suppression, Auto Gain
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: { ideal: true },
          noiseSuppression: { ideal: true },
          autoGainControl: { ideal: true },
          channelCount: 1,
          sampleRate: { ideal: 16000 },
        },
      });
      audioStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        const ctx = new AudioCtx({ sampleRate: 16000 });
        audioContextRef.current = ctx;

        const source = ctx.createMediaStreamSource(stream);

        // 1. High-Pass Filter: cuts fan vibration, desk rumble, and electrical hum below 85Hz
        const highPass = ctx.createBiquadFilter();
        highPass.type = "highpass";
        highPass.frequency.value = 85;
        highPass.Q.value = 0.7;

        // 2. Vocal Peaking Filter: boosts human voice presence band (1.5kHz - 3.5kHz)
        const vocalPresence = ctx.createBiquadFilter();
        vocalPresence.type = "peaking";
        vocalPresence.frequency.value = 2000;
        vocalPresence.Q.value = 1.0;
        vocalPresence.gain.value = 3.0;

        // 3. Analyser for Voice Activity Detection (VAD) / Noise floor gating
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 512;
        audioAnalyserRef.current = analyser;

        source.connect(highPass);
        highPass.connect(vocalPresence);
        vocalPresence.connect(analyser);
      }
      return stream;
    } catch (err) {
      console.warn("[AudioPipeline] Could not start hardware audio filter, using direct recognizer:", err);
      return null;
    }
  }, []);

  // Voice Activity Detection: ensures input contains genuine speech energy above ambient noise floor
  const hasVoiceEnergy = useCallback(() => {
    const analyser = audioAnalyserRef.current;
    if (!analyser) return true;
    const buffer = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(buffer);
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      const normalized = (buffer[i] - 128) / 128;
      sum += normalized * normalized;
    }
    const rms = Math.sqrt(sum / buffer.length);
    // Cuts laptop fan noise/idle hum while catching natural speech
    return rms > 0.012;
  }, []);

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
      stopAudioPipeline();
    };

    recognizer.onerror = (e) => {
      console.debug("Speech recognition error:", e.error);
      setIsListening(false);
      stopAudioPipeline();
    };

    recognizer.onresult = (e) => {
      let currentTranscript = "";
      for (let i = 0; i < e.results.length; i++) {
        currentTranscript += e.results[i][0].transcript + " ";
      }
      currentTranscript = currentTranscript.trim();

      if (!currentTranscript) return;

      // Filter phantom noise syllables caused by fan drone, breath, or room echo
      const lower = currentTranscript.toLowerCase();
      if (/^(uh|um|ah|er|eh|oh|\.|\,)$/.test(lower)) return;

      // Discard low-energy phantom activations
      if (!hasVoiceEnergy() && currentTranscript.length < 3) return;

      setVoiceTranscript(currentTranscript);
      setSearchMode("voice");

      // Instant local lexical match as words are spoken
      const instantHits = lexicalSearch(currentTranscript);
      if (instantHits.length > 0) {
        setFilteredEntries(instantHits);
      }

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
          } else if (instantHits.length === 0) {
            setFilteredEntries([]);
          }
        } finally {
          setIsSearching(false);
        }
      }, 50);
    };

    recognizerRef.current = recognizer;

    return () => {
      stopAudioPipeline();
    };
  }, [hasVoiceEnergy, stopAudioPipeline]);

  const toggleListening = async () => {
    if (isListening) {
      setIsListening(false);
      try {
        recognizerRef.current?.stop();
      } catch {}

      setIsTranscribing(true);
      try {
        // Transcribe noise-robust 16kHz PCM audio with Moonshine
        const moonshineText = await stopAudioCaptureAndTranscribe();
        const clean = (moonshineText || voiceTranscript).trim();

        if (clean) {
          setVoiceTranscript(clean);
          setSearchMode("voice");
          setIsSearching(true);
          try {
            const bestHit = await fastSearch(clean);
            if (bestHit) {
              setFilteredEntries(Array.isArray(bestHit) ? bestHit : [bestHit]);
            } else {
              setFilteredEntries([]);
            }
          } finally {
            setIsSearching(false);
          }
        }
      } catch (err) {
        console.warn("[Moonshine] STT pass error, falling back to interim transcript:", err);
      } finally {
        setIsTranscribing(false);
      }
    } else {
      setIsListening(true);
      setVoiceTranscript("");
      try {
        // Start hardware-level noise reduction and 16kHz audio buffer capture
        await startAudioCapture();
        try {
          recognizerRef.current?.start();
        } catch {
          // WebSpeech is optional live preview helper
        }
      } catch (err) {
        console.error("[Moonshine] Failed to activate microphone:", err);
        setIsListening(false);
        cancelAudioCapture();
      }
    }
  };

  const clearVoiceSearch = () => {
    setVoiceTranscript("");
    setSearchMode("all");
    setFilteredEntries(allEntries.slice(0, 20));
    cancelAudioCapture();
  };

  // ── Live feed filtering for manual text search (Instant zero-wait + non-blocking refinement)
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

      // 1. Instant local search filtering: 0ms latency, never waits for model downloads
      const instantHits = lexicalSearch(q);
      setFilteredEntries(instantHits.slice(0, 20));
      setIsSearching(false);

      // 2. Non-blocking vector re-ranking if model is ready
      searchDebounceRef.current = setTimeout(async () => {
        try {
          const ranked = await rankedSearch(q);
          if (ranked && ranked.length > 0) {
            setFilteredEntries(ranked.slice(0, 20));
          }
        } catch (err) {
          console.debug("[Search] Vector enhancement skipped:", err);
        }
      }, 75);
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

    // Immediate instant results
    const instantHits = lexicalSearch(query);
    if (instantHits.length > 0) {
      setFilteredEntries(instantHits.slice(0, 20));
    }
    setIsSearching(false);

    try {
      const ranked = await rankedSearch(query);
      if (ranked && ranked.length > 0) {
        setFilteredEntries(ranked.slice(0, 20));
      }
    } catch (err) {
      console.debug("[Search] Submit fallback:", err);
    }
  };

  // ── Quick note with semantic auto-tagging ───────────────────────────────────
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

  // ── In-card editing ─────────────────────────────────────────────────────────
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

  // ── Card Voice Note & Quick Labeling Secondary Fallback ─────────────────────
  const stopCardVoiceLabel = useCallback(() => {
    if (cardRecognizerRef.current) {
      try {
        cardRecognizerRef.current.stop();
      } catch {}
      cardRecognizerRef.current = null;
    }
    setRecordingCardId(null);
  }, []);

  const handleStartVoiceLabel = useCallback((entry, e) => {
    if (e) e.stopPropagation();

    if (recordingCardId === entry.id) {
      stopCardVoiceLabel();
      return;
    }

    stopCardVoiceLabel();

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setQuickLabelingId(entry.id);
      setQuickLabelInput(entry.label || "");
      return;
    }

    try {
      const recognizer = new SpeechRecognition();
      recognizer.continuous = false;
      recognizer.interimResults = false;
      recognizer.lang = "en-US";

      recognizer.onstart = () => {
        setRecordingCardId(entry.id);
      };

      recognizer.onerror = (event) => {
        console.warn("[VoiceLabel] Speech recognition error:", event.error);
        stopCardVoiceLabel();
      };

      recognizer.onend = () => {
        setRecordingCardId(null);
      };

      recognizer.onresult = async (event) => {
        let transcript = "";
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript + " ";
        }
        transcript = transcript.trim();
        if (!transcript) return;

        const normalized = transcript.toLowerCase().replace(/[^a-z0-9]/g, "_");
        const derived = deriveTags(transcript);
        const labelTags = [`#label_${normalized}`, `#${normalized}`, ...derived];
        const existingTags = entry.tags || [];
        const combinedTags = Array.from(new Set([...existingTags, ...labelTags, "#photo"]));

        await db.images.update(entry.id, {
          label: transcript,
          tags: combinedTags,
          clipFailed: false,
        });

        const updated = await refreshCache();
        const sorted = [...updated].sort((a, b) => b.timestamp - a.timestamp);
        setAllEntries(sorted);
        setFilteredEntries((curr) =>
          curr.map((item) =>
            item.id === entry.id
              ? { ...item, label: transcript, tags: combinedTags, clipFailed: false }
              : item
          )
        );
        if (selectedEntry?.id === entry.id) {
          setSelectedEntry((curr) =>
            curr ? { ...curr, label: transcript, tags: combinedTags, clipFailed: false } : curr
          );
        }
        stopCardVoiceLabel();
      };

      cardRecognizerRef.current = recognizer;
      recognizer.start();
    } catch (err) {
      console.warn("[VoiceLabel] Could not start speech recognizer:", err);
      setQuickLabelingId(entry.id);
      setQuickLabelInput(entry.label || "");
    }
  }, [recordingCardId, stopCardVoiceLabel, selectedEntry?.id]);

  const handleSaveQuickLabel = useCallback(async (entry, labelText, e) => {
    if (e) e.stopPropagation();
    const cleanLabel = (labelText || quickLabelInput).trim();
    if (!cleanLabel) return;

    const normalized = cleanLabel.toLowerCase().replace(/[^a-z0-9]/g, "_");
    const derived = deriveTags(cleanLabel);
    const labelTags = [`#label_${normalized}`, `#${normalized}`, ...derived];
    const existingTags = entry.tags || [];
    const combinedTags = Array.from(new Set([...existingTags, ...labelTags, "#photo"]));

    await db.images.update(entry.id, {
      label: cleanLabel,
      tags: combinedTags,
      clipFailed: false,
    });

    const updated = await refreshCache();
    const sorted = [...updated].sort((a, b) => b.timestamp - a.timestamp);
    setAllEntries(sorted);
    setFilteredEntries((curr) =>
      curr.map((item) =>
        item.id === entry.id
          ? { ...item, label: cleanLabel, tags: combinedTags, clipFailed: false }
          : item
      )
    );
    if (selectedEntry?.id === entry.id) {
      setSelectedEntry((curr) =>
        curr ? { ...curr, label: cleanLabel, tags: combinedTags, clipFailed: false } : curr
      );
    }

    setQuickLabelingId(null);
    setQuickLabelInput("");
  }, [quickLabelInput, selectedEntry?.id]);

  // ── Image Capture / Vision & OCR Pipeline ────────────────────────────────────
  const extractTagsFromText = (text) => {
    if (!text || !text.trim()) return ["#image", "#photo"];
    const tags = deriveTags(text);
    return tags.length > 0 ? Array.from(new Set([...tags, "#image"])) : ["#image", "#photo"];
  };

  const handleImageCapture = async (e) => {
    const rawFiles = Array.from(e.target.files || []);
    if (rawFiles.length === 0) return;

    if (fileInputRef.current) fileInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";

    try {
      const newEntries = rawFiles.map((file, i) => {
        const id = createId();
        const instantUrl = URL.createObjectURL(file);
        setImageThumbnailUrl(id, instantUrl);
        return {
          id,
          type: "image",
          imageBlob: file,
          thumbnailUrl: instantUrl,
          label: "",
          ocr: "",
          isPureImage: true,
          tags: ["#image", "#photo"],
          timestamp: Date.now() + i,
        };
      });
      newEntries.forEach((entry) => thumbnailJobsRef.current.add(entry.id));

      await db.images.bulkAdd(newEntries);
      const updated = await refreshCache();
      const sorted = [...updated].sort((a, b) => b.timestamp - a.timestamp);
      setAllEntries(sorted);
      setFilteredEntries(sorted.slice(0, 20));

      // Run OCR & CLIP Vector Embedding silently in background
      const processingTask = (async () => {
        for (let i = 0; i < newEntries.length; i++) {
          const entry = newEntries[i];
          const file = rawFiles[i];
          if (!file) continue;

          try {
            console.log(`[Vision] Embedding image ${i + 1}/${newEntries.length}...`);

            // Generate compressed thumbnail for IndexedDB storage in background
            createImageThumbnail(file)
              .then(async (thumbnail) => {
                if (thumbnail) {
                  await db.images.update(entry.id, { thumbnailBlob: thumbnail });
                }
              })
              .catch((err) => console.warn("[Vision] Thumbnail background generation:", err));

            const isValidTag = (tag) => {
              const clean = tag.replace("#", "").toLowerCase();
              if (clean.length < 3 || !/[aeiou0-9]/.test(clean) || /(.)\1\1/.test(clean)) return false;
              return !["npn", "wwn", "broic", "ji", "er", "te"].includes(clean);
            };

            let currentCardTags = ["#photo"];

            const syncCard = async (patch) => {
              if (patch.tags && Array.isArray(patch.tags)) {
                const set = new Set([...currentCardTags, ...patch.tags]);
                currentCardTags = Array.from(set)
                  .filter(isValidTag)
                  .sort((a, b) => (a === "#photo" ? 1 : b === "#photo" ? -1 : 0));
                patch.tags = currentCardTags;
              }
              await db.images.update(entry.id, patch);
              const updateEntry = (item) => item.id === entry.id ? { ...item, ...patch } : item;
              setFilteredEntries((curr) => curr.map(updateEntry));
              setAllEntries((curr) => curr.map(updateEntry));
            };

            let currentOcrText = "";
            let currentPlate = null;

            // Run OCR (Tesseract worker) and CLIP embedding (Vision worker) concurrently
            const ocrPromise = (async () => {
              try {
                const ocrResult = await performOCR(file);
                const extractedText = typeof ocrResult === "string" ? ocrResult : (ocrResult?.text || "");
                currentPlate = typeof ocrResult === "object" && ocrResult?.plate ? ocrResult.plate : null;
                const structuredIds = typeof ocrResult === "object" && Array.isArray(ocrResult?.structuredIds) ? ocrResult.structuredIds : [];
                const cleanText = extractedText.trim();
                currentOcrText = currentPlate && !cleanText.includes(currentPlate)
                  ? `${currentPlate}\n${cleanText}`.trim()
                  : cleanText;

                const textTags = cleanText ? deriveTags(cleanText) : [];
                const idTags = structuredIds.map((id) => `#${id.value.toLowerCase()}`);
                const plateTags = currentPlate ? ["#vehicle", `#${currentPlate.replace(/\s+/g, "").toLowerCase()}`] : [];
                const docTags = structuredIds.length > 0 ? ["#document"] : [];

                const ocrTags = Array.from(new Set([
                  ...textTags,
                  ...plateTags,
                  ...docTags,
                  ...idTags,
                  "#photo",
                ])).filter(isValidTag);

                // Update the UI card immediately with OCR results as soon as ready
                await syncCard({
                  ocr: currentOcrText,
                  isPureImage: !currentPlate && structuredIds.length === 0 && (currentOcrText.length === 0 || isPureImageText(currentOcrText)),
                  tags: ocrTags.length > 0 ? ocrTags : ["#photo"],
                });
              } catch (ocrErr) {
                console.warn("[Vision] OCR detection pass failed:", ocrErr);
              }
            })();

            const embedPromise = (async () => {
              try {
                const embedResult = await embedImage(file);
                const embedding = Array.isArray(embedResult)
                  ? embedResult
                  : embedResult?.embedding;
                const visualTags = Array.isArray(embedResult?.visualTags)
                  ? embedResult.visualTags
                  : [];

                const patch = {};
                if (embedding && Array.isArray(embedding)) {
                  patch.embedding = embedding;
                }
                if (visualTags.length > 0) {
                  patch.tags = visualTags;
                }
                if (Object.keys(patch).length > 0) {
                  await syncCard(patch);
                }
              } catch (embedErr) {
                console.warn("[Vision] CLIP embedding pass failed:", embedErr);
              }
            })();

            await Promise.all([ocrPromise, embedPromise]);

            // Secondary fallback check: did CLIP fail to classify this image?
            // (e.g. keys inside bag, small/occluded items where CLIP produced no specific tags)
            const updatedRecord = await db.images.get(entry.id);
            const currentTags = updatedRecord?.tags || [];
            const hasSpecificTags = currentTags.some((t) => t !== "#photo" && t !== "#image" && t !== "#picture");
            const hasPlateOrText = Boolean(updatedRecord?.ocr?.trim() && !isPureImageText(updatedRecord?.ocr));
            const clipFailed = !updatedRecord?.label && !hasSpecificTags && !hasPlateOrText;

            if (clipFailed) {
              await db.images.update(entry.id, { clipFailed: true });
              const markFailed = (item) => item.id === entry.id ? { ...item, clipFailed: true } : item;
              setFilteredEntries((curr) => curr.map(markFailed));
              setAllEntries((curr) => curr.map(markFailed));
            }

          } catch (err) {
            console.error("[Vision] Image processing failed:", err);
          } finally {
            thumbnailJobsRef.current.delete(entry.id);
          }
        }

        const refreshed = await refreshCache();
        const refreshedSorted = [...refreshed].sort((a, b) => b.timestamp - a.timestamp);
        setAllEntries(refreshedSorted);

        try {
          if (query.trim()) {
            const ranked = await rankedSearch(query);
            setFilteredEntries(ranked.slice(0, 20));
          } else {
            setFilteredEntries(refreshedSorted.slice(0, 20));
          }
        } catch (err) {
          console.error("[Vision] Search refresh failed:", err);
        }
      })();

      imageProcessingQueueRef.current = processingTask.catch((err) => {
        console.error("[Vision] Background image processing queue failed:", err);
        newEntries.forEach((entry) => thumbnailJobsRef.current.delete(entry.id));
      });

    } catch (err) {
      console.error("Real-time image upload failed:", err);
      alert("Error adding images. Please try again.");
    }
  };

  const handleDeleteEntry = async (entry, e) => {
    if (e) e.stopPropagation();

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

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 flex flex-col items-center px-4 py-6 antialiased selection:bg-indigo-500 selection:text-white relative overflow-x-hidden">
      {/* Ambient background glows */}
      <div className="fixed top-[-140px] left-1/2 -translate-x-1/2 w-[520px] h-[360px] bg-gradient-to-b from-indigo-600/20 via-violet-600/10 to-transparent blur-[110px] pointer-events-none rounded-full" />
      <div className="fixed bottom-[-120px] right-[-100px] w-[360px] h-[360px] bg-sky-500/10 blur-[130px] pointer-events-none rounded-full" />

      {/* Clean Mobile App Header */}
      <header className="w-full max-w-md flex items-center justify-between pb-4 mb-2 z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <span className="text-white text-sm">✦</span>
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight text-white/95">
              VoicePad
            </h1>
            <p className="text-[11px] text-slate-400 font-normal">
              Instant memory capture
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400/80 shadow-[0_0_8px_rgba(52,211,153,0.6)]" />
        </div>
      </header>

      <main className="w-full max-w-md flex flex-col items-center z-10">
        {/* Soft Ambient Voice Orb */}
        <div className="relative my-4 flex flex-col items-center">
          {/* Conversational Floater Bubble */}
          {(voiceTranscript || isListening) && (
            <div className="absolute -top-11 z-20 flex flex-col items-center animate-in fade-in zoom-in-95 duration-200">
              <div className="px-3.5 py-1.5 bg-slate-900/90 text-white rounded-full shadow-2xl border border-white/10 flex items-center gap-2 backdrop-blur-xl">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
                <span className="text-xs font-medium tracking-tight truncate max-w-[200px]">
                  {voiceTranscript ? `"${voiceTranscript}"` : "Listening..."}
                </span>
                {voiceTranscript && (
                  <button
                    onClick={clearVoiceSearch}
                    className="text-slate-400 hover:text-white text-xs px-1 hover:bg-white/10 rounded-full transition"
                    title="Clear voice query"
                    aria-label="Clear voice query"
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Organic Glow */}
          <div
            className={`absolute inset-0 rounded-full blur-2xl transition-all duration-500 pointer-events-none ${
              isListening
                ? "bg-rose-500/40 scale-125 animate-pulse"
                : isTranscribing
                ? "bg-amber-500/30 scale-110"
                : "bg-indigo-500/20 scale-100"
            }`}
          />

          {/* Glowing Voice Button */}
          <button
            onClick={toggleListening}
            disabled={isTranscribing}
            className={`relative z-10 w-24 h-24 rounded-full flex flex-col items-center justify-center transition-all duration-300 shadow-2xl active:scale-95 select-none ${
              isTranscribing
                ? "bg-gradient-to-tr from-amber-600 to-amber-500 shadow-amber-500/30 ring-2 ring-white/20"
                : isListening
                ? "bg-gradient-to-tr from-rose-600 to-pink-600 shadow-rose-500/40 ring-4 ring-rose-400/30 scale-105"
                : "bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-500 hover:brightness-110 shadow-indigo-500/30 ring-1 ring-white/20"
            }`}
            aria-label={isTranscribing ? "Processing voice" : isListening ? "Stop listening" : "Tap to speak"}
          >
            {isListening ? (
              <div className="flex items-center gap-1 my-1">
                <span className="w-1 h-3.5 bg-white rounded-full animate-[bounce_1s_infinite_100ms]" />
                <span className="w-1 h-5 bg-white rounded-full animate-[bounce_1s_infinite_200ms]" />
                <span className="w-1 h-3 bg-white rounded-full animate-[bounce_1s_infinite_300ms]" />
              </div>
            ) : isTranscribing ? (
              <span className="text-xs font-medium text-white/95 my-1">Thinking</span>
            ) : (
              <svg className="w-6 h-6 text-white my-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 02-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            )}
            <span className="text-[10px] font-medium text-white/80">
              {isTranscribing ? "Resolving" : isListening ? "Tap to finish" : "Tap to speak"}
            </span>
          </button>
        </div>

        {/* Ambient Action Pills */}
        <div className="w-full flex items-center justify-center gap-2.5 my-2">
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
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-2xl text-xs font-medium bg-white/[0.05] hover:bg-white/[0.08] border border-white/[0.08] text-white/90 backdrop-blur-md transition active:scale-95 shadow-sm"
          >
            <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span>Capture Photo</span>
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-2xl text-xs font-medium bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.06] text-slate-300 backdrop-blur-md transition active:scale-95 shadow-sm"
          >
            <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span>From Library</span>
          </button>
        </div>

        {/* Ambient Search Bar */}
        <form onSubmit={handleSearchSubmit} className="w-full mt-2">
          <div className="relative flex items-center">
            <span className="absolute left-3.5 text-slate-400 pointer-events-none">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </span>
            <input
              type="text"
              placeholder="Search memories, photos, notes..."
              value={query}
              onChange={handleQueryChange}
              className="w-full pl-10 pr-9 py-2.5 bg-white/[0.04] border border-white/[0.08] hover:border-white/[0.12] rounded-2xl text-xs text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500/50 backdrop-blur-xl transition shadow-inner"
            />
            {query && (
              <button
                type="button"
                onClick={() => {
                  setQuery("");
                  setSearchMode("all");
                  setFilteredEntries(allEntries.slice(0, 20));
                }}
                className="absolute right-3 text-slate-400 hover:text-white text-xs transition"
                aria-label="Clear search"
              >
                ✕
              </button>
            )}
          </div>
          {query.trim() && searchMode === "text" && (
            <p className="text-[11px] text-slate-400 mt-1.5 ml-2">
              {filteredEntries.length} {filteredEntries.length === 1 ? "match" : "matches"} found
            </p>
          )}
        </form>

        {/* Quick Note Input */}
        <form onSubmit={handleAddQuickNote} className="w-full mt-2">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Remember a quick thought..."
              value={manualNote}
              onChange={handleNoteChange}
              className="flex-1 px-3.5 py-2 bg-white/[0.03] border border-white/[0.06] rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/40 transition"
            />
            <button
              type="submit"
              className="px-3.5 py-2 bg-white/[0.06] hover:bg-white/[0.1] text-slate-200 rounded-xl text-xs font-medium border border-white/[0.08] transition whitespace-nowrap active:scale-95"
            >
              Save
            </button>
          </div>
          {manualNote.trim() && autoTags.length > 0 && (
            <div className="flex items-center gap-1.5 mt-2 flex-wrap animate-in fade-in duration-150">
              {autoTags.map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 font-medium"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </form>

        {/* Ambient Feed */}
        <section className="w-full mt-6">
          <div className="flex items-center justify-between mb-3 px-1">
            <h2 className="text-xs font-medium text-slate-400">
              {searchMode === "voice" && voiceTranscript.trim() ? (
                <span className="flex items-center gap-1.5 text-white/90">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Voice Match
                </span>
              ) : query.trim() ? (
                `${filteredEntries.length} ${filteredEntries.length === 1 ? "match" : "matches"}`
              ) : (
                "Recent Memories"
              )}
            </h2>
            {searchMode === "voice" && voiceTranscript.trim() ? (
              <button
                onClick={clearVoiceSearch}
                className="text-[11px] text-indigo-400 hover:text-indigo-300 transition"
              >
                Show all
              </button>
            ) : null}
          </div>

          {isSearching ? (
            <div className="space-y-2.5 animate-pulse" aria-label="Loading search results">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="p-3.5 bg-white/[0.02] border border-white/[0.04] rounded-2xl flex items-start justify-between gap-3"
                >
                  <div className="flex-1 space-y-2">
                    <div className="h-3.5 bg-white/[0.06] rounded w-3/4" />
                    <div className="h-3 bg-white/[0.04] rounded w-1/2" />
                  </div>
                  <div className="w-10 h-3 bg-white/[0.04] rounded shrink-0" />
                </div>
              ))}
            </div>
          ) : filteredEntries.length === 0 && (query.trim() || voiceTranscript.trim()) ? (
            <div className="py-12 text-center">
              <p className="text-slate-400 text-sm">
                No memories found for "{voiceTranscript || query}"
              </p>
              <p className="text-slate-500 text-xs mt-1">
                Try a different phrase or add a label to your photo.
              </p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {filteredEntries.map((entry) => {
                const isEditing = editingId === entry.id;
                const isPureImage =
                  entry.type === "image" &&
                  (entry.isPureImage === true || (!entry.ocr && !entry.label) || isPureImageText(entry.ocr));
                const isClipFailed =
                  entry.type === "image" &&
                  (entry.clipFailed === true ||
                    (!entry.label &&
                      (!entry.tags ||
                        entry.tags.length === 0 ||
                        entry.tags.every((t) => t === "#photo" || t === "#image" || t === "#picture")) &&
                      (!entry.ocr || !entry.ocr.trim() || isPureImageText(entry.ocr))));

                return (
                  <div
                    key={entry.id}
                    onClick={() => {
                      if (!isEditing) {
                        setOcrExpandedId(null);
                        setSelectedImage(null);
                        setSelectedEntry(entry);
                      }
                    }}
                    className="p-3.5 bg-white/[0.03] hover:bg-white/[0.05] border border-white/[0.06] hover:border-white/[0.12] rounded-2xl transition flex flex-col gap-2.5 shadow-sm cursor-pointer backdrop-blur-md"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                          {entry.type === "image" && !isPureImage && entry.ocr && (
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation();
                                setOcrExpandedId((currentId) => currentId === entry.id ? null : entry.id);
                              }}
                              aria-expanded={ocrExpandedId === entry.id}
                              className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-white/[0.05] hover:bg-white/[0.1] text-slate-300 border border-white/[0.08] transition flex items-center gap-1"
                            >
                              <span>📄</span>
                              <span>{ocrExpandedId === entry.id ? "Hide text" : "Show text"}</span>
                            </button>
                          )}
                          {entry.label ? (
                            <span className="text-[10px] font-medium px-2.5 py-0.5 rounded-full bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 flex items-center gap-1">
                              🏷️ {entry.label}
                            </span>
                          ) : entry.type === "image" ? (
                            <div className="flex items-center gap-1.5">
                              <button
                                onClick={(e) => startEditing(entry, e)}
                                className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] text-slate-300 transition"
                              >
                                + Label
                              </button>
                              <button
                                type="button"
                                onClick={(e) => handleStartVoiceLabel(entry, e)}
                                className={`text-[10px] font-medium px-2 py-0.5 rounded-full border transition flex items-center gap-1 ${
                                  recordingCardId === entry.id
                                    ? "bg-rose-500/20 text-rose-300 border-rose-500/40 animate-pulse"
                                    : "bg-white/[0.04] hover:bg-white/[0.08] text-slate-300 border-white/[0.08]"
                                }`}
                              >
                                🎤 {recordingCardId === entry.id ? "Listening..." : "Voice Note"}
                              </button>
                            </div>
                          ) : null}
                        </div>

                        {/* Inline Edit Form OR Card Content */}
                        {isEditing ? (
                          <div className="mt-1 space-y-2">
                            {entry.type === "image" && (
                              <div>
                                <label className="text-[10px] text-slate-400 block mb-1 font-medium">
                                  Label:
                                </label>
                                <input
                                  type="text"
                                  value={editLabel}
                                  onChange={(e) => setEditLabel(e.target.value)}
                                  className="w-full px-3 py-1.5 bg-slate-900 border border-white/10 rounded-xl text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 mb-2"
                                  placeholder="e.g. Blue bag with keys, ID Card..."
                                  autoFocus
                                />
                              </div>
                            )}
                            <textarea
                              value={editText}
                              onChange={(e) => setEditText(e.target.value)}
                              rows={isPureImage ? 2 : 3}
                              className="w-full p-2.5 bg-slate-900 border border-white/10 rounded-xl text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                              placeholder={
                                isPureImage
                                  ? "Description or notes..."
                                  : entry.type === "image"
                                  ? "Extracted text..."
                                  : "Edit memory content..."
                              }
                            />
                            <div className="flex items-center gap-2 pt-1">
                              <button
                                onClick={(e) => handleSaveEdit(entry, e)}
                                className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-medium transition shadow"
                              >
                                Save
                              </button>
                              <button
                                onClick={cancelEditing}
                                className="px-3.5 py-1.5 bg-white/[0.06] hover:bg-white/[0.1] text-slate-300 rounded-xl text-xs font-medium transition"
                              >
                                Cancel
                              </button>
                            </div>
                            {editingError && (
                              <p className="text-[11px] text-rose-400 mt-1 font-medium">
                                {editingError}
                              </p>
                            )}
                          </div>
                        ) : entry.type === "image" ? (
                          <div className="mt-1 space-y-2">
                            <div className="w-full aspect-[4/3] max-h-80 rounded-2xl overflow-hidden bg-black/40 border border-white/[0.06] flex items-center justify-center p-1">
                              {entry.thumbnailUrl ? (
                                <img
                                  src={entry.thumbnailUrl}
                                  alt={entry.label || "Memory photo"}
                                  loading="lazy"
                                  decoding="async"
                                  fetchPriority="low"
                                  className="max-h-72 max-w-full object-contain rounded-xl"
                                />
                              ) : (
                                <span className="px-4 text-center text-xs text-slate-500">
                                  {entry.thumbnailFailed ? "Tap to open photo" : "Loading..."}
                                </span>
                              )}
                            </div>
                            {!isPureImage && ocrExpandedId === entry.id && entry.ocr && (
                              <p
                                id={`ocr-text-${entry.id}`}
                                className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap bg-white/[0.02] p-2.5 rounded-xl border border-white/[0.04]"
                              >
                                {entry.ocr}
                              </p>
                            )}
                            <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                              {entry.tags?.map((tag) => (
                                <span
                                  key={tag}
                                  className="text-[10px] px-2.5 py-0.5 rounded-full bg-white/[0.04] text-slate-300 border border-white/[0.06] font-medium"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>

                            {/* Human-friendly ambient prompt when image is unclassified */}
                            {isClipFailed && (
                              <div
                                onClick={(e) => e.stopPropagation()}
                                className="p-3 rounded-2xl bg-white/[0.02] border border-indigo-500/20 text-slate-200 flex flex-col gap-2 mt-2"
                              >
                                <p className="text-[11px] text-slate-300 leading-relaxed">
                                  Need help finding this later? Add a 2-second voice note or label.
                                </p>

                                {quickLabelingId === entry.id ? (
                                  <div className="space-y-2 mt-1">
                                    <div className="flex gap-1.5">
                                      <input
                                        type="text"
                                        value={quickLabelInput}
                                        onChange={(e) => setQuickLabelInput(e.target.value)}
                                        onKeyDown={(e) => {
                                          if (e.key === "Enter") {
                                            e.preventDefault();
                                            handleSaveQuickLabel(entry, quickLabelInput, e);
                                          }
                                        }}
                                        placeholder="e.g. Keys inside bag zip..."
                                        className="flex-1 px-3 py-1.5 bg-slate-900 border border-white/10 rounded-xl text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                                        autoFocus
                                      />
                                      <button
                                        type="button"
                                        onClick={(e) => handleSaveQuickLabel(entry, quickLabelInput, e)}
                                        className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-xl text-xs transition"
                                      >
                                        Save
                                      </button>
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setQuickLabelingId(null);
                                        }}
                                        className="px-2.5 py-1.5 bg-white/[0.06] hover:bg-white/[0.1] text-slate-300 rounded-xl text-xs transition"
                                      >
                                        Cancel
                                      </button>
                                    </div>

                                    <div className="flex items-center gap-1.5 flex-wrap pt-0.5">
                                      {["Keys in bag", "Car keys", "House keys", "Backpack", "Wallet", "Receipt"].map((chip) => (
                                        <button
                                          key={chip}
                                          type="button"
                                          onClick={(e) => handleSaveQuickLabel(entry, chip, e)}
                                          className="text-[10px] px-2.5 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] text-indigo-300 border border-white/[0.06] transition"
                                        >
                                          + {chip}
                                        </button>
                                      ))}
                                    </div>
                                  </div>
                                ) : (
                                  <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                                    <button
                                      type="button"
                                      onClick={(e) => handleStartVoiceLabel(entry, e)}
                                      className={`px-3 py-1.5 rounded-xl text-xs font-medium flex items-center gap-1.5 transition ${
                                        recordingCardId === entry.id
                                          ? "bg-rose-500 text-white animate-pulse"
                                          : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm"
                                      }`}
                                    >
                                      <span>{recordingCardId === entry.id ? "⏹️ Stop" : "🎤 2-Sec Voice Note"}</span>
                                      {recordingCardId === entry.id && (
                                        <span className="text-[10px] text-white/90">Listening...</span>
                                      )}
                                    </button>

                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setQuickLabelingId(entry.id);
                                        setQuickLabelInput("");
                                      }}
                                      className="px-3 py-1.5 bg-white/[0.05] hover:bg-white/[0.08] text-slate-200 border border-white/[0.08] rounded-xl text-xs font-medium transition flex items-center gap-1"
                                    >
                                      <span>🏷️ Add Label</span>
                                    </button>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        ) : (
                          <>
                            <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                              {entry.content}
                            </p>
                            <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                              {entry.tags?.map((tag) => (
                                <span
                                  key={tag}
                                  className="text-[10px] px-2.5 py-0.5 rounded-full bg-white/[0.04] text-slate-300 border border-white/[0.06] font-medium"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </>
                        )}
                      </div>

                      {!isEditing && (
                        <div className="flex flex-col items-end gap-1.5 shrink-0">
                          <span className="text-[10px] text-slate-500">
                            {formatTime(entry.timestamp)}
                          </span>
                          <div className="flex items-center gap-1">
                            <button
                              onClick={(e) => startEditing(entry, e)}
                              className="text-[11px] text-slate-400 hover:text-white px-1.5 py-0.5 rounded-lg hover:bg-white/[0.06] transition"
                              title="Edit memory"
                              aria-label="Edit memory"
                            >
                              Edit
                            </button>
                            <button
                              onClick={(e) => handleDeleteEntry(entry, e)}
                              className="text-[11px] text-slate-400 hover:text-rose-400 px-1.5 py-0.5 rounded-lg hover:bg-white/[0.06] transition"
                              title="Delete memory"
                              aria-label="Delete memory"
                            >
                              Delete
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

      {/* Centered zoomed memory modal */}
      {selectedEntry && (() => {
        const modalEntry = selectedEntry;
        const modalPure =
          modalEntry.type === "image" &&
          (modalEntry.isPureImage === true || (!modalEntry.ocr && !modalEntry.label) || isPureImageText(modalEntry.ocr));
        const modalTags = modalEntry.tags || [];

        return (
          <div
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-2xl flex items-center justify-center p-4 animate-in fade-in duration-150"
            onClick={() => setSelectedEntry(null)}
          >
            <div
              className="w-full max-w-2xl max-h-[90vh] overflow-y-auto p-4 bg-slate-900/90 border border-white/10 rounded-3xl shadow-2xl animate-in zoom-in-95 duration-150"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-1.5 flex-wrap">
                  {modalEntry.type === "image" && !modalPure && modalEntry.ocr && (
                    <button
                      type="button"
                      onClick={() => setOcrExpandedId((currentId) => currentId === modalEntry.id ? null : modalEntry.id)}
                      aria-expanded={ocrExpandedId === modalEntry.id}
                      className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-white/[0.06] hover:bg-white/[0.1] text-slate-300 border border-white/[0.08] transition flex items-center gap-1"
                    >
                      <span>📄</span>
                      <span>{ocrExpandedId === modalEntry.id ? "Hide text" : "Show text"}</span>
                    </button>
                  )}
                  {modalEntry.label ? (
                    <span className="text-[10px] font-medium px-2.5 py-0.5 rounded-full bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 flex items-center gap-1">
                      🏷️ {modalEntry.label}
                    </span>
                  ) : modalEntry.type === "image" ? (
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => { startEditing(modalEntry); setSelectedEntry(null); }}
                        className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] text-slate-300 transition"
                      >
                        + Add Label
                      </button>
                      <button
                        type="button"
                        onClick={(e) => handleStartVoiceLabel(modalEntry, e)}
                        className={`text-[10px] font-medium px-2 py-0.5 rounded-full border transition flex items-center gap-1 ${
                          recordingCardId === modalEntry.id
                            ? "bg-rose-500/20 text-rose-300 border-rose-500/40 animate-pulse"
                            : "bg-white/[0.04] hover:bg-white/[0.08] text-slate-300 border-white/[0.08]"
                        }`}
                      >
                        🎤 {recordingCardId === modalEntry.id ? "Listening..." : "Voice Note"}
                      </button>
                    </div>
                  ) : null}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-500">
                    {formatTime(modalEntry.timestamp)}
                  </span>
                  <button
                    onClick={() => setSelectedEntry(null)}
                    className="text-slate-400 hover:text-white text-sm p-1 rounded-full hover:bg-white/10 transition"
                    aria-label="Close modal"
                  >
                    ✕
                  </button>
                </div>
              </div>

              {modalEntry.type === "image" ? (
                <div className="mt-2 space-y-2">
                  <div className="w-full max-h-[70vh] rounded-2xl overflow-hidden bg-black/60 border border-white/[0.08] flex items-center justify-center p-1">
                    {selectedImage?.id === modalEntry.id ? (
                      <img
                        src={selectedImage.url}
                        alt={modalEntry.label || "Memory photo"}
                        loading="eager"
                        decoding="async"
                        className="max-h-[68vh] w-full object-contain rounded-xl"
                      />
                    ) : (
                      <span className="text-sm text-slate-400 py-8">Loading image...</span>
                    )}
                  </div>
                  {!modalPure && ocrExpandedId === modalEntry.id && modalEntry.ocr && (
                    <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap bg-white/[0.03] p-3 rounded-2xl border border-white/[0.06]">
                      {modalEntry.ocr}
                    </p>
                  )}
                </div>
              ) : (
                <div className="mt-2">
                  <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                    {modalEntry.content}
                  </p>
                </div>
              )}

              {modalTags?.length > 0 && (
                <div className="flex items-center gap-1.5 mt-3 flex-wrap">
                  {modalTags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] px-2.5 py-0.5 rounded-full bg-white/[0.04] text-slate-300 border border-white/[0.06] font-medium"
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
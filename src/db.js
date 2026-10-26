// src/db.js
import Dexie from "dexie";
import { createId } from "./id.js";

export const db = new Dexie("voicepad-db");

db.version(1).stores({
  notes: "id, type, *tags, timestamp",
  images: "id, type, blobUrl, ocr, *tags, timestamp",
});

/** Seed demo data strictly once without generating duplicate cards */
export async function seedDemoData() {
  const noteCount = await db.notes.count();
  const imgCount = await db.images.count();
  
  if (noteCount === 0 && imgCount === 0) {
    const now = Date.now();
    await db.notes.bulkAdd([
      {
        id: createId(),
        type: "text",
        content: "Call John at 5 pm about the insurance",
        tags: ["#reminder"],
        timestamp: now - 2 * 60 * 60 * 1000,
      },
      {
        id: createId(),
        type: "text",
        content: "Parking spot #42, license plate ABC123",
        tags: ["#vehicle"],
        timestamp: now - 30 * 60 * 1000,
      },
    ]);
    console.log("Demo data seeded");
  } else {
    // Deduplicate any repeated demo notes that were seeded in previous bugged runs
    const allNotes = await db.notes.toArray();
    const seen = new Set();
    for (const note of allNotes) {
      if (seen.has(note.content)) {
        await db.notes.delete(note.id);
      } else {
        seen.add(note.content);
      }
    }
  }
}

export default db;

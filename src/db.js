// src/db.js
import Dexie from "dexie";

export const db = new Dexie("voicepad-db");

db.version(1).stores({
  notes: "id, type, *tags, timestamp",
  images: "id, type, blobUrl, ocr, *tags, timestamp",
});

/** Seed demo data if database is empty */
export async function seedDemoData() {
  const noteCount = await db.notes.count();
  const imgCount = await db.images.count();
  if (noteCount === 0 && imgCount === 0) {
    const now = Date.now();
    await db.notes.bulkAdd([
      {
        id: crypto.randomUUID(),
        type: "text",
        content: "Call John at 5 pm about the insurance",
        tags: ["#reminder"],
        timestamp: now - 2 * 60 * 60 * 1000,
      },
      {
        id: crypto.randomUUID(),
        type: "text",
        content: "Parking spot #42, license plate ABC123",
        tags: ["#vehicle"],
        timestamp: now - 30 * 60 * 1000,
      },
    ]);
    console.log("Demo data seeded");
  }
}

export default db;

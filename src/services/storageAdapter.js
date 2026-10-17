import { Capacitor } from "@capacitor/core";
import { Filesystem, Directory } from "@capacitor/filesystem";

/**
 * Saves large image files directly to the phone's native storage
 * to avoid browser IndexedDB quota eviction on low-memory mobile devices.
 */
export async function saveImageToDisk(fileName, base64Data) {
  if (Capacitor.isNativePlatform()) {
    const savedFile = await Filesystem.writeFile({
      path: `voicepad_images/${fileName}`,
      data: base64Data,
      directory: Directory.Data,
      recursive: true,
    });
    return savedFile.uri;
  }
  // On browser, store directly in IndexedDB via Dexie
  return base64Data;
}
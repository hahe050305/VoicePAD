import { Capacitor } from "@capacitor/core";
import { Camera, CameraResultType, CameraSource } from "@capacitor/camera";

/**
 * Captures a photo:
 * - On Native Phone: Opens direct OS hardware camera viewfinder, resizes in native layer
 * - On Laptop / Browser: Triggers standard browser file input dialog
 */
export async function capturePhotoNativeOrWeb(fallbackFileInputRef = null) {
  if (Capacitor.isNativePlatform()) {
    try {
      const photo = await Camera.getPhoto({
        quality: 85,
        width: 1600,
        allowEditing: false,
        correctOrientation: true,
        resultType: CameraResultType.Uri,
        source: CameraSource.Camera,
      });
      if (photo?.webPath) {
        const res = await fetch(photo.webPath);
        return await res.blob();
      }
      return null;
    } catch (err) {
      if (err?.message !== "User cancelled photos app" && err?.message !== "User cancelled") {
        console.warn("[CameraAdapter] Native camera failed, falling back to file input:", err);
      }
      return null;
    }
  } else {
    // 💻 Laptop / Browser Fallback
    if (fallbackFileInputRef && fallbackFileInputRef.current) {
      fallbackFileInputRef.current.click();
    }
    return null; // Handled asynchronously via input onChange
  }
}
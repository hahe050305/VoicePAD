import { Capacitor } from "@capacitor/core";
import { Camera, CameraResultType, CameraSource } from "@capacitor/camera";

/**
 * Captures a photo:
 * - On Native Phone: Opens direct OS hardware camera viewfinder
 * - On Laptop / Browser: Triggers standard browser file input dialog
 */
export async function capturePhotoNativeOrWeb(fallbackFileInputRef = null) {
  if (Capacitor.isNativePlatform()) {
    // 📱 Native Phone Execution
    const photo = await Camera.getPhoto({
      quality: 90,
      allowEditing: false,
      resultType: CameraResultType.DataUrl,
      source: CameraSource.Camera, // CameraSource.Prompt allows user to pick Camera vs Photos
    });
    return photo.dataUrl; // Ready for Tesseract OCR / IndexedDB
  } else {
    // 💻 Laptop / Browser Fallback
    if (fallbackFileInputRef && fallbackFileInputRef.current) {
      fallbackFileInputRef.current.click();
    }
    return null; // Handled asynchronously via input onChange
  }
}
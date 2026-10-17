import { Capacitor } from "@capacitor/core";
import { VoiceRecorder } from "capacitor-voice-recorder";

/**
 * Microphone manager:
 * - On Native Phone: Uses capacitor-voice-recorder with OS hardware audio permissions
 * - On Laptop / Browser: Uses standard Web SpeechRecognition / getUserMedia
 */
export const voiceAdapter = {
  isNative: Capacitor.isNativePlatform(),

  async requestPermission() {
    if (Capacitor.isNativePlatform()) {
      const { value } = await VoiceRecorder.requestAudioRecordingPermission();
      return value;
    }
    return true; // Browser handles this on mic activation
  },

  async startRecording() {
    if (Capacitor.isNativePlatform()) {
      const hasPermission = await VoiceRecorder.hasAudioRecordingPermission();
      if (!hasPermission.value) {
        await VoiceRecorder.requestAudioRecordingPermission();
      }
      return await VoiceRecorder.startRecording();
    }
    // Web fallback handled by existing Web SpeechRecognition / Web Audio
  },

  async stopRecording() {
    if (Capacitor.isNativePlatform()) {
      const result = await VoiceRecorder.stopRecording();
      // result.value contains: { recordDataBase64, mimeType, msDuration }
      return result.value;
    }
  },
};
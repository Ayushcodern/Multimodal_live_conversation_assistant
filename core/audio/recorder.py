"""
Microphone Audio Recorder Module.

Handles audio capture directly from physical input devices using sounddevice.
Automatically queries and validates the Windows OS / PortAudio default input device.
Captures real acoustic samples in memory without creating persistent files.
"""

import io
import threading
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd


class AudioRecorder:
    """Manages recording of real microphone input from the Windows default device in-memory."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Audio sampling frequency in Hz (default 16000Hz, optimal for STT).
            channels: Number of audio channels (1 for mono).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._lock = threading.Lock()
        self._is_recording = False
        self._recorded_chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._active_device_index: Optional[int] = None

    @property
    def is_recording(self) -> bool:
        """Return True if microphone is actively capturing audio."""
        with self._lock:
            return self._is_recording

    @staticmethod
    def get_default_input_device() -> Tuple[bool, Optional[int], str, str]:
        """
        Dynamically detects and validates the current Windows / PortAudio default input device.

        Returns:
            Tuple[bool, Optional[int], str, str]:
                (is_valid, device_index, device_name, status_message)
        """
        try:
            # Query the default device indices [input_device_index, output_device_index]
            default_indices = sd.default.device
            default_in_idx = default_indices[0] if default_indices else -1

            if default_in_idx is None or default_in_idx < 0:
                return (
                    False,
                    None,
                    "No Default Device Detected",
                    "No default recording device is configured in Windows. "
                    "Please connect or enable a microphone and set it as the default input device in Windows Sound Settings.",
                )

            # Query the device details directly by its PortAudio index
            device_info = sd.query_devices(default_in_idx)
            max_in_channels = device_info.get("max_input_channels", 0)

            if max_in_channels <= 0:
                return (
                    False,
                    default_in_idx,
                    str(device_info.get("name", "Unknown Device")),
                    f"The Windows default input device ('{device_info.get('name')}') does not support audio input channels. "
                    "Please check your Windows Sound Settings.",
                )

            device_name = str(device_info.get("name", f"Microphone (Index {default_in_idx})"))
            return (
                True,
                default_in_idx,
                device_name,
                f"Ready ({device_name})",
            )

        except Exception as e:
            return (
                False,
                None,
                "Device Query Error",
                f"Unable to query Windows default audio device: {str(e)}. "
                "Ensure audio services are running and your microphone is connected.",
            )

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback invoked by sounddevice for each audio block."""
        if self._is_recording:
            self._recorded_chunks.append(indata.copy())

    def start_recording(self) -> Tuple[bool, str]:
        """
        Start recording audio from the validated Windows default microphone.
        Re-checks the default device on each invocation.

        Returns:
            Tuple[bool, str]: (Success status, descriptive message).
        """
        with self._lock:
            if self._is_recording:
                return True, "Recording is already active."

            # Re-check the OS default device dynamically
            is_valid, dev_idx, dev_name, check_msg = self.get_default_input_device()
            if not is_valid or dev_idx is None:
                return False, (
                    f"{check_msg}\n\n"
                    "If access is blocked, check: "
                    "Windows Settings → Privacy & security → Microphone → "
                    "enable 'Microphone access' and 'Let desktop apps access your microphone'."
                )

            self._active_device_index = dev_idx
            self._recorded_chunks.clear()

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    device=self._active_device_index,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_recording = True
                return True, f"Recording started on Windows default device: '{dev_name}'."
            except Exception as e:
                self._is_recording = False
                self._stream = None
                self._active_device_index = None
                return False, (
                    f"Microphone access error: {str(e)}.\n"
                    "Please check Windows Settings → Privacy & security → Microphone → "
                    "ensure 'Microphone access' and 'Let desktop apps access your microphone' are both enabled."
                )

    def stop_recording(self) -> Tuple[bool, Optional[bytes], str]:
        """
        Stop microphone recording and return genuine captured audio as in-memory WAV bytes.

        Returns:
            Tuple[bool, Optional[bytes], str]:
                (Success status, WAV audio bytes or None, status message).
        """
        with self._lock:
            if not self._is_recording or self._stream is None:
                return False, None, "No active recording to stop."

            self._is_recording = False
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None
                self._active_device_index = None

            if not self._recorded_chunks:
                return False, None, "No audio data was captured."

            # Concatenate captured chunks
            audio_data = np.concatenate(self._recorded_chunks, axis=0)
            self._recorded_chunks.clear()

            # Encode as standard in-memory WAV
            buffer = io.BytesIO()
            wavfile.write(buffer, self.sample_rate, audio_data)
            wav_bytes = buffer.getvalue()
            duration = len(audio_data) / self.sample_rate

            return True, wav_bytes, f"Captured {duration:.1f}s of real audio."

    def release(self):
        """Safely release audio stream resources."""
        with self._lock:
            self._is_recording = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                self._active_device_index = None
            self._recorded_chunks.clear()

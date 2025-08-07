import wave
import time
import threading
import numpy as np
import tempfile
import os
from typing import Optional
from dataclasses import dataclass

try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import noisereduce as nr

    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False


@dataclass
class AudioConfig:
    """Audio recording configuration"""
    sample_rate: int = 16000  # 16kHz optimal for Whisper
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
    energy_threshold: int = 500
    noise_reduction: bool = True
    normalize_audio: bool = True


class WhisperSpeechRecognition:
    """Optimized Whisper-only speech recognition"""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size
        self.config = AudioConfig()
        self.audio = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
        self.stream = None
        self.is_recording = False
        self.audio_buffer = []
        self.recording_thread = None

        # Load model on init for faster first transcription
        self._load_model()

    def _load_model(self):
        """Load Whisper model with error handling"""
        if not WHISPER_AVAILABLE:
            print("Whisper not installed! Install with: pip install openai-whisper")
            return False

        try:
            print(f"Loading Whisper {self.model_size} model...")
            # Use English-only model for faster performance
            model_name = f"{self.model_size}.en"
            self.model = whisper.load_model(model_name)
            print(f"Whisper {model_name} model loaded successfully")
            return True
        except Exception as e:
            # Fallback to multilingual model
            try:
                print(f"English model failed, loading multilingual {self.model_size}...")
                self.model = whisper.load_model(self.model_size)
                print(f"Whisper {self.model_size} model loaded")
                return True
            except Exception as e2:
                print(f"Failed to load Whisper model: {e2}")
                return False

    def set_model_size(self, size: str):
        """Change model size if needed"""
        if size != self.model_size:
            self.model_size = size
            self._load_model()

    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """Start audio recording"""
        if not PYAUDIO_AVAILABLE or not self.audio:
            return False

        try:
            self.stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.chunk_size
            )

            self.is_recording = True
            self.audio_buffer = []

            self.recording_thread = threading.Thread(target=self._record_continuous)
            self.recording_thread.start()

            return True
        except Exception as e:
            print(f"Recording start failed: {e}")
            return False

    def _record_continuous(self):
        """Record audio continuously"""
        while self.is_recording:
            try:
                data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
                self.audio_buffer.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break

    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return audio data"""
        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join(timeout=0.5)

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None

        if self.audio_buffer:
            return b''.join(self.audio_buffer)
        return None

    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """Apply noise reduction and normalization"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # Noise reduction
            if self.config.noise_reduction and NOISE_REDUCE_AVAILABLE:
                audio_float = audio_np.astype(np.float32) / 32768.0
                audio_float = nr.reduce_noise(
                    y=audio_float,
                    sr=self.config.sample_rate,
                    stationary=True,
                    prop_decrease=1.0
                )
                audio_np = (audio_float * 32768.0).astype(np.int16)

            # Normalization
            if self.config.normalize_audio:
                max_val = np.max(np.abs(audio_np))
                if max_val > 0:
                    audio_np = (audio_np * (0.9 * 32768 / max_val)).astype(np.int16)

            return audio_np.tobytes()
        except Exception as e:
            print(f"Audio preprocessing error: {e}")
            return audio_data

    def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using Whisper"""
        if not audio_data or not self.model:
            return None

        # Preprocess
        processed_audio = self.preprocess_audio(audio_data)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        try:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(processed_audio)

            # Transcribe with optimized settings
            result = self.model.transcribe(
                temp_file.name,
                language='en',
                fp16=False,
                verbose=False,
                # Performance optimizations
                beam_size=5,  # Default is 5, lower = faster
                best_of=5,  # Default is 5
                temperature=0,  # Single temp instead of range for speed
                condition_on_previous_text=False,  # Faster
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )

            text = result["text"].strip()
            return text if text else None

        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return None
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()


# Simple factory function
def create_enhanced_recognizer(config_manager):
    """Create Whisper speech recognizer"""
    model_size = config_manager.get("whisper_model_size", "base")

    # For speed: tiny > base > small > medium > large
    # For accuracy: large > medium > small > base > tiny
    recognizer = WhisperSpeechRecognition(model_size)

    # Apply config
    recognizer.config.energy_threshold = config_manager.get("mic_energy_threshold", 500)
    recognizer.config.noise_reduction = config_manager.get("noise_reduction", True)
    recognizer.config.normalize_audio = config_manager.get("normalize_audio", True)

    return recognizer
import io
import wave
import time
import threading
import queue
import numpy as np
import tempfile
import os
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# Try to import various speech recognition libraries
try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import azure.cognitiveservices.speech as speechsdk

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import assemblyai as aai

    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False

try:
    import whisper

    WHISPER_LOCAL_AVAILABLE = True
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False

try:
    from scipy import signal
    import noisereduce as nr

    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False


class SpeechProvider(Enum):
    """Available speech recognition providers"""
    GOOGLE = "google"  # Free, poor accuracy
    WHISPER_API = "whisper_api"  # OpenAI Whisper API - high accuracy
    WHISPER_LOCAL = "whisper_local"  # Local Whisper model (probably best choice lol)
    AZURE = "azure"  # Azure Speech Services
    ASSEMBLYAI = "assemblyai"  # AssemblyAI API


@dataclass
class AudioConfig:
    """Audio recording configuration"""
    sample_rate: int = 16000  # 16kHz is good for speech
    channels: int = 1  # Mono is sufficient for speech
    chunk_size: int = 1024  # Buffer size
    format: int = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
    energy_threshold: int = 500  # Microphone sensitivity
    pause_threshold: float = 0.8  # Seconds of silence before stopping
    noise_reduction: bool = True  # Apply noise reduction
    normalize_audio: bool = True  # Normalize audio levels


class ContinuousAudioRecorder:
    """Continuous audio recorder that captures complete audio"""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.audio = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
        self.stream = None
        self.is_recording = False
        self.audio_buffer = []
        self.recording_thread = None

    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """Start continuous audio recording"""
        if not PYAUDIO_AVAILABLE or not self.audio:
            print("PyAudio not available")
            return False

        try:
            # Open audio stream
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

            # Start recording thread
            self.recording_thread = threading.Thread(target=self._record_continuous)
            self.recording_thread.start()

            return True

        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False

    def _record_continuous(self):
        """Continuously record audio while is_recording is True"""
        while self.is_recording:
            try:
                # Read audio chunk
                data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
                self.audio_buffer.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break

    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return the complete audio data"""
        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=0.5)

        # Close stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None

        # Combine all audio chunks
        if self.audio_buffer:
            return b''.join(self.audio_buffer)
        return None

    def __del__(self):
        """Cleanup"""
        if self.audio:
            self.audio.terminate()


class EnhancedSpeechRecognition:
    """Enhanced speech recognition with multiple backend support"""

    def __init__(self, provider: SpeechProvider = SpeechProvider.GOOGLE):
        self.provider = provider
        self.recorder = ContinuousAudioRecorder()
        self.config = AudioConfig()

        # Initialize provider-specific clients
        self.openai_client = None
        self.azure_config = None
        self.whisper_model = None
        self.assemblyai_client = None

        # Audio processing
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None

    def initialize_provider(self, **kwargs):
        """Initialize the selected speech provider with credentials"""
        if self.provider == SpeechProvider.WHISPER_API:
            if OPENAI_AVAILABLE and 'api_key' in kwargs:
                self.openai_client = OpenAI(api_key=kwargs['api_key'])
                print("Whisper API initialized")

        elif self.provider == SpeechProvider.WHISPER_LOCAL:
            if WHISPER_LOCAL_AVAILABLE:
                model_size = kwargs.get('model_size', 'base')
                print(f"Loading Whisper {model_size} model...")
                self.whisper_model = whisper.load_model(model_size)
                print("Whisper local model loaded")

        elif self.provider == SpeechProvider.AZURE:
            if AZURE_AVAILABLE and 'subscription_key' in kwargs:
                self.azure_config = speechsdk.SpeechConfig(
                    subscription=kwargs['subscription_key'],
                    region=kwargs.get('region', 'eastus')
                )
                print("Azure Speech initialized")

        elif self.provider == SpeechProvider.ASSEMBLYAI:
            if ASSEMBLYAI_AVAILABLE and 'api_key' in kwargs:
                aai.settings.api_key = kwargs['api_key']
                self.assemblyai_client = aai.Transcriber()
                print("AssemblyAI initialized")

    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """Apply noise reduction and normalization to audio"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio_data

        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # Apply noise reduction
            if self.config.noise_reduction:
                # Simple noise reduction using scipy
                audio_float = audio_np.astype(np.float32) / 32768.0
                # Estimate noise from first 0.5 seconds
                noise_sample_size = min(int(self.config.sample_rate * 0.5), len(audio_float))
                if noise_sample_size > 0:
                    audio_float = nr.reduce_noise(
                        y=audio_float,
                        sr=self.config.sample_rate,
                        stationary=True
                    )
                audio_np = (audio_float * 32768.0).astype(np.int16)

            # Normalize audio levels
            if self.config.normalize_audio:
                max_val = np.max(np.abs(audio_np))
                if max_val > 0:
                    # Normalize to 80% of max to avoid clipping
                    audio_np = (audio_np * (0.8 * 32768 / max_val)).astype(np.int16)

            return audio_np.tobytes()

        except Exception as e:
            print(f"Audio preprocessing error: {e}")
            return audio_data

    def save_audio_to_wav(self, audio_data: bytes) -> str:
        """Save audio data to a temporary WAV file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')

        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_data)

        return temp_file.name

    def transcribe_audio(self, audio_data: bytes, language: str = "en") -> Optional[str]:
        """Transcribe audio using the selected provider"""
        if not audio_data:
            return None

        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_data)

        try:
            if self.provider == SpeechProvider.WHISPER_API:
                return self._transcribe_whisper_api(processed_audio, language)
            elif self.provider == SpeechProvider.WHISPER_LOCAL:
                return self._transcribe_whisper_local(processed_audio, language)
            elif self.provider == SpeechProvider.AZURE:
                return self._transcribe_azure(processed_audio, language)
            elif self.provider == SpeechProvider.ASSEMBLYAI:
                return self._transcribe_assemblyai(processed_audio, language)
            else:  # Default to Google
                return self._transcribe_google(processed_audio, language)

        except Exception as e:
            print(f"Transcription error with {self.provider.value}: {e}")
            # Fallback to Google if available
            if self.provider != SpeechProvider.GOOGLE and SR_AVAILABLE:
                print("Falling back to Google Speech Recognition...")
                return self._transcribe_google(processed_audio, language)
            return None

    def _transcribe_whisper_api(self, audio_data: bytes, language: str) -> Optional[str]:
        """Transcribe using OpenAI Whisper API"""
        if not self.openai_client:
            print("Whisper API client not initialized")
            return None

        # Save audio to temporary file
        temp_file = self.save_audio_to_wav(audio_data)

        try:
            with open(temp_file, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            return transcript.strip()

        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _transcribe_whisper_local(self, audio_data: bytes, language: str) -> Optional[str]:
        """Transcribe using local Whisper model"""
        if not self.whisper_model:
            print("Whisper local model not loaded")
            return None

        # Save audio to temporary file
        temp_file = self.save_audio_to_wav(audio_data)

        try:
            result = self.whisper_model.transcribe(
                temp_file,
                language=language,
                fp16=False  # Use FP32 for better compatibility
            )
            return result["text"].strip()

        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _transcribe_azure(self, audio_data: bytes, language: str) -> Optional[str]:
        """Transcribe using Azure Speech Services"""
        if not self.azure_config:
            print("Azure Speech not configured")
            return None

        # Create audio stream
        stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=stream)

        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.azure_config,
            audio_config=audio_config
        )

        # Push audio data
        stream.write(audio_data)
        stream.close()

        # Recognize
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text.strip()
        else:
            print(f"Azure recognition failed: {result.reason}")
            return None

    def _transcribe_assemblyai(self, audio_data: bytes, language: str) -> Optional[str]:
        """Transcribe using AssemblyAI"""
        if not self.assemblyai_client:
            print("AssemblyAI not configured")
            return None

        # Save audio to temporary file
        temp_file = self.save_audio_to_wav(audio_data)

        try:
            transcript = self.assemblyai_client.transcribe(temp_file)
            if transcript.status == aai.TranscriptStatus.completed:
                return transcript.text.strip()
            else:
                print(f"AssemblyAI transcription failed: {transcript.error}")
                return None

        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _transcribe_google(self, audio_data: bytes, language: str) -> Optional[str]:
        """Transcribe using Google Speech Recognition (fallback)"""
        if not SR_AVAILABLE or not self.recognizer:
            print("Speech recognition not available")
            return None

        try:
            # Convert to AudioData object
            audio = sr.AudioData(
                audio_data,
                self.config.sample_rate,
                2  # 16-bit audio = 2 bytes per sample
            )

            # Try recognition with different parameters
            text = self.recognizer.recognize_google(
                audio,
                language=f"{language}-US",
                show_all=False
            )

            return text.strip()

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Google Speech Recognition error: {e}")
            return None

    def record_and_transcribe(
            self,
            device_index: Optional[int] = None,
            max_duration: float = 30.0,
            callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """Complete recording and transcription flow"""

        # Start recording
        if not self.recorder.start_recording(device_index):
            return None

        print(f"Recording started (max {max_duration}s)...")

        # Record for specified duration or until stopped
        start_time = time.time()
        while self.recorder.is_recording and (time.time() - start_time) < max_duration:
            time.sleep(0.1)

        # Stop recording
        audio_data = self.recorder.stop_recording()

        if not audio_data:
            print("No audio recorded")
            return None

        print(f"Recorded {len(audio_data) / self.config.sample_rate / 2:.1f} seconds of audio")

        # Transcribe
        print(f"Transcribing with {self.provider.value}...")
        text = self.transcribe_audio(audio_data)

        if text and callback:
            callback(text)

        return text


# Example integration function for the main app
def create_enhanced_recognizer(config_manager) -> EnhancedSpeechRecognition:
    """Factory function to create the best available speech recognizer"""

    # Determine best provider based on available APIs
    if config_manager.get("openai_api_key") and OPENAI_AVAILABLE:
        provider = SpeechProvider.WHISPER_API
        recognizer = EnhancedSpeechRecognition(provider)
        recognizer.initialize_provider(api_key=config_manager.get("openai_api_key"))
        print("Using OpenAI Whisper API for speech recognition")

    elif config_manager.get("azure_tts_key") and AZURE_AVAILABLE:
        provider = SpeechProvider.AZURE
        recognizer = EnhancedSpeechRecognition(provider)
        recognizer.initialize_provider(
            subscription_key=config_manager.get("azure_tts_key"),
            region=config_manager.get("azure_tts_region", "eastus")
        )
        print("Using Azure Speech Services for speech recognition")

    elif WHISPER_LOCAL_AVAILABLE:
        provider = SpeechProvider.WHISPER_LOCAL
        recognizer = EnhancedSpeechRecognition(provider)
        recognizer.initialize_provider(model_size="base")  # or "tiny" for faster
        print("Using local Whisper model for speech recognition")

    else:
        # Fallback to Google
        provider = SpeechProvider.GOOGLE
        recognizer = EnhancedSpeechRecognition(provider)
        print("Using Google Speech Recognition (free tier)")

    # Configure audio settings
    recognizer.config.energy_threshold = config_manager.get("mic_energy_threshold", 500)
    recognizer.config.noise_reduction = config_manager.get("noise_reduction", True)
    recognizer.config.normalize_audio = config_manager.get("normalize_audio", True)

    return recognizer
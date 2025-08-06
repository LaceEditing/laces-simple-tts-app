"""
Lace's Simple TTS App - AI Streamer Assistant with Twitch Integration
Beautiful lavender-themed UI with complete streaming features
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, font
import json
import os
import threading
import time
import queue
import base64
from pathlib import Path
from PIL import Image, ImageTk
import keyboard
import pyautogui
import cv2
import numpy as np
import requests
import tempfile
import shutil
import socket
import re
from datetime import datetime
import sys
import io
import wave

# Optional imports (will check if available)
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from elevenlabs import client as elevenlabs_client, VoiceSettings

    ELEVENLABS_AVAILABLE = True
except ImportError:
    try:
        from elevenlabs import generate, save, set_api_key, voices

        ELEVENLABS_AVAILABLE = True
        ELEVENLABS_OLD_API = True
    except ImportError:
        ELEVENLABS_AVAILABLE = False
        ELEVENLABS_OLD_API = False

try:
    import azure.cognitiveservices.speech as speechsdk

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import gtts

    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Constants
CONFIG_FILE = "config.json"
HISTORY_FILE = "conversation_history.json"
OBS_OUTPUT_FILE = "current_avatar.png"  # File that OBS will monitor
ICON_FILE = "app_icon.ico"  # Application icon file

# Beautiful Lavender Color Scheme
COLORS = {
    'bg': '#E8E0F7',  # Light lavender background
    'frame_bg': '#F5F0FA',  # Even lighter for frames
    'accent': '#9B59B6',  # Purple accent
    'button': '#A569BD',  # Button purple
    'button_hover': '#8E44AD',  # Darker purple on hover
    'text': '#2C3E50',  # Dark text
    'entry_bg': '#FFFFFF',  # White entry background
    'success': '#27AE60',  # Green for success
    'error': '#E74C3C',  # Red for errors
    'info': '#3498DB',  # Blue for info
    'twitch': '#6441A4'  # Twitch purple
}

# Voice providers and their available voices
VOICE_OPTIONS = {
    "streamelements": {
        "name": "StreamElements (Free)",
        "voices": {
            "Brian": "Brian",
            "Amy": "Amy",
            "Emma": "Emma",
            "Geraint": "Geraint",
            "Russell": "Russell",
            "Nicole": "Nicole",
            "Joey": "Joey",
            "Justin": "Justin",
            "Matthew": "Matthew",
            "Ivy": "Ivy",
            "Joanna": "Joanna",
            "Kendra": "Kendra",
            "Kimberly": "Kimberly",
            "Salli": "Salli"
        }
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "voices": {
            "Adam": "Adam",
            "Antoni": "Antoni",
            "Arnold": "Arnold",
            "Bella": "Bella",
            "Domi": "Domi",
            "Elli": "Elli",
            "Josh": "Josh",
            "Rachel": "Rachel",
            "Sam": "Sam"
        }
    },
    "azure": {
        "name": "Azure TTS",
        "voices": {
            "en-US-JennyNeural": "Jenny (Female)",
            "en-US-GuyNeural": "Guy (Male)",
            "en-US-AriaNeural": "Aria (Female)",
            "en-US-DavisNeural": "Davis (Male)",
            "en-US-AmberNeural": "Amber (Female)",
            "en-US-AshleyNeural": "Ashley (Female)",
            "en-US-BrandonNeural": "Brandon (Male)",
            "en-US-ChristopherNeural": "Christopher (Male)",
            "en-US-CoraNeural": "Cora (Female)",
            "en-US-ElizabethNeural": "Elizabeth (Female)",
            "en-US-EricNeural": "Eric (Male)",
            "en-US-JacobNeural": "Jacob (Male)",
            "en-US-MichelleNeural": "Michelle (Female)",
            "en-US-MonicaNeural": "Monica (Female)",
            "en-US-RogerNeural": "Roger (Male)"
        }
    },
    "gtts": {
        "name": "Google TTS (Free)",
        "voices": {
            "en": "English",
            "en-au": "English (Australia)",
            "en-uk": "English (UK)",
            "en-us": "English (US)",
            "en-ca": "English (Canada)",
            "en-in": "English (India)"
        }
    },
    "pyttsx3": {
        "name": "System TTS (Offline)",
        "voices": {}  # Will be populated dynamically
    }
}

DEFAULT_CONFIG = {
    "openai_api_key": "",
    "elevenlabs_api_key": "",
    "azure_tts_key": "",
    "azure_tts_region": "eastus",
    "twitch_username": "",
    "twitch_oauth": "",
    "twitch_channel": "",
    "voice_provider": "streamelements",
    "voice_name": "Brian",
    "idle_image": "idle.png",
    "speaking_image": "speaking.png",
    "system_prompt": "You are a helpful, albeit sassy individual watching a stream and interacting with the streamer and chat.",
    "push_to_talk_key": "v",
    "auto_commentary": False,
    "commentary_interval": 15,
    "respond_to_twitch": True,
    "twitch_response_chance": 0.8,
    "obs_mode": True,
    "openai_model": "gpt-4o-mini",
    "twitch_cooldown": 5,  # Seconds between Twitch responses
    "twitch_read_username": True,  # Read username before message
    "twitch_read_message": True,  # Read the message content
    "twitch_ai_response": True,  # Generate AI response
    "input_device_index": -1,  # Default microphone
    "output_device_index": -1,  # Default speakers
    "ai_memory_messages": 20,  # Number of messages to remember
    "ai_max_tokens": 150,  # Max tokens for AI response
    "ai_temperature": 0.7,  # AI creativity level
    "save_conversation_history": True,  # Save conversation to file
    "elevenlabs_stability": 0.5,
    "elevenlabs_similarity": 0.75
}


class ConversationHistory:
    """Manages conversation history persistence"""

    def __init__(self, filename=HISTORY_FILE):
        self.filename = filename
        self.history = self.load_history()
        self.max_entries = 1000  # Maximum entries to keep in file

    def load_history(self):
        """Load conversation history from file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def add_entry(self, role, content, metadata=None):
        """Add an entry to the conversation history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.history.append(entry)

        # Trim history if too long
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]

        self.save_history()
        return entry

    def save_history(self):
        """Save conversation history to file"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation history: {e}")

    def get_recent(self, count=10):
        """Get recent conversation entries"""
        return self.history[-count:] if self.history else []

    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.save_history()


class ConfigManager:
    """Handles loading and saving configuration"""

    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    config = DEFAULT_CONFIG.copy()
                    config.update(loaded_config)
                    return config
            except:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def get(self, key, default=None):
        """Get a configuration value"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a configuration value and save"""
        self.config[key] = value
        return self.save_config()


class AudioDeviceManager:
    """Manages audio input/output device selection"""

    @staticmethod
    def get_audio_devices():
        """Get list of available audio devices"""
        input_devices = []
        output_devices = []

        if PYAUDIO_AVAILABLE:
            try:
                p = pyaudio.PyAudio()

                print(f"Found {p.get_device_count()} audio devices:")

                # Get device info
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    device_info = {
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'] or info['maxOutputChannels']
                    }

                    print(
                        f"  Device {i}: {info['name']} (In:{info['maxInputChannels']}, Out:{info['maxOutputChannels']})")

                    # Categorize as input or output
                    if info['maxInputChannels'] > 0:
                        input_devices.append(device_info)
                    if info['maxOutputChannels'] > 0:
                        output_devices.append(device_info)

                p.terminate()

                print(f"Found {len(input_devices)} input devices and {len(output_devices)} output devices")
            except Exception as e:
                print(f"Error getting audio devices: {e}")

        return input_devices, output_devices

    @staticmethod
    def set_pygame_output_device(device_index):
        """Set pygame output device"""
        if PYGAME_AVAILABLE and device_index >= 0:
            try:
                pygame.mixer.quit()
                # Pygame doesn't support specific device selection easily
                # This is a placeholder - actual implementation would need platform-specific code
                pygame.mixer.init(frequency=48000, buffer=1024)
                return True
            except Exception as e:
                print(f"Error setting output device: {e}")
                return False
        return False


class TwitchChat:
    """Handles Twitch IRC connection and chat messages"""

    def __init__(self, config_manager, message_callback=None):
        self.config = config_manager
        self.socket = None
        self.connected = False
        self.message_callback = message_callback
        self.last_response_time = 0
        self.connection_thread = None

    def connect(self):
        """Connect to Twitch IRC"""
        username = self.config.get("twitch_username")
        oauth = self.config.get("twitch_oauth")
        channel = self.config.get("twitch_channel")

        if not all([username, oauth, channel]):
            return False, "Missing Twitch credentials. Please fill in all fields."

        # Clean up oauth token format
        oauth = oauth.strip()
        if not oauth.startswith("oauth:"):
            oauth = f"oauth:{oauth}"

        try:
            print(f"Connecting to Twitch as {username} to channel #{channel}...")

            # Create socket connection
            self.socket = socket.socket()
            self.socket.settimeout(5.0)  # Add timeout for connection
            self.socket.connect(('irc.chat.twitch.tv', 6667))

            # Send authentication
            self.socket.send(f"PASS {oauth}\r\n".encode('utf-8'))
            self.socket.send(f"NICK {username}\r\n".encode('utf-8'))

            # Wait for authentication response
            response = self.socket.recv(1024).decode('utf-8')
            print(f"Auth response: {response}")

            if "Welcome" not in response and "001" not in response:
                if "Login authentication failed" in response:
                    return False, "Authentication failed. Check your OAuth token."

            # Join channel
            self.socket.send(f"JOIN #{channel}\r\n".encode('utf-8'))

            # Request capabilities
            self.socket.send("CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership\r\n".encode('utf-8'))

            # Check join response
            response = self.socket.recv(1024).decode('utf-8')
            print(f"Join response: {response}")

            self.connected = True
            self.socket.settimeout(0.1)  # Set non-blocking for reading

            # Send a test message to confirm we're connected
            if self.message_callback:
                self.message_callback("system", f"✅ Connected to #{channel}!")

            return True, "Successfully connected to Twitch!"

        except socket.timeout:
            return False, "Connection timed out. Check your internet connection."
        except Exception as e:
            print(f"Twitch connection error: {e}")
            return False, f"Connection error: {str(e)}"

    def disconnect(self):
        """Disconnect from Twitch IRC"""
        self.connected = False
        if self.socket:
            try:
                self.socket.send("QUIT\r\n".encode('utf-8'))
                self.socket.close()
            except:
                pass
        self.socket = None

        if self.message_callback:
            self.message_callback("system", "❌ Disconnected from Twitch")

    def send_message(self, message):
        """Send a message to Twitch chat (if needed for responses)"""
        if not self.connected or not self.socket:
            return False

        channel = self.config.get("twitch_channel")
        try:
            self.socket.send(f"PRIVMSG #{channel} :{message}\r\n".encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def read_chat(self):
        """Read messages from Twitch chat"""
        if not self.connected or not self.socket:
            return []

        messages = []
        try:
            response = self.socket.recv(2048).decode('utf-8', errors='ignore')

            # Handle PING to stay connected
            if response.startswith('PING'):
                self.socket.send("PONG\r\n".encode('utf-8'))
                return []

            # Parse messages
            for line in response.split('\r\n'):
                if line:
                    # Parse PRIVMSG (chat messages)
                    if 'PRIVMSG' in line:
                        try:
                            # Extract username and message
                            username_match = re.search(r':(\w+)!', line)
                            message_match = re.search(r'PRIVMSG #\w+ :(.+)', line)

                            if username_match and message_match:
                                username = username_match.group(1)
                                message = message_match.group(1)
                                messages.append({
                                    'username': username,
                                    'message': message,
                                    'timestamp': datetime.now()
                                })

                                # Send to callback if provided
                                if self.message_callback:
                                    self.message_callback("twitch_chat", f"{username}: {message}")
                        except Exception as e:
                            print(f"Error parsing message: {e}")

                    # Handle JOIN/PART messages for user awareness
                    elif 'JOIN' in line and not line.startswith(':'):
                        username_match = re.search(r':(\w+)!', line)
                        if username_match and self.message_callback:
                            self.message_callback("system", f"👋 {username_match.group(1)} joined")

                    elif 'PART' in line:
                        username_match = re.search(r':(\w+)!', line)
                        if username_match and self.message_callback:
                            self.message_callback("system", f"👋 {username_match.group(1)} left")

        except socket.timeout:
            # This is normal, just means no new messages
            pass
        except Exception as e:
            if self.connected:  # Only print error if we think we're connected
                print(f"Error reading chat: {e}")

        return messages


class StreamElementsTTS:
    """StreamElements TTS API integration"""

    @staticmethod
    def generate_speech(text, voice="Brian"):
        """Generate speech using StreamElements API"""
        url = "https://api.streamelements.com/kappa/v2/speech"
        params = {"voice": voice, "text": text}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.write(response.content)
                temp_file.close()
                return temp_file.name
            else:
                print(f"StreamElements TTS error: {response.status_code}")
                return None
        except Exception as e:
            print(f"StreamElements TTS error: {e}")
            return None


class TTSManager:
    """Manages text-to-speech with multiple backends"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.engine = None
        self.elevenlabs_client = None
        self.azure_config = None
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.init_tts()
        self.start_speech_processor()

    def init_tts(self):
        """Initialize TTS engine based on availability"""
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=48000, buffer=1024)

        # Initialize pyttsx3 if available
        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)

                # Populate system voices
                voices = self.engine.getProperty('voices')
                VOICE_OPTIONS["pyttsx3"]["voices"] = {}
                for voice in voices:
                    # Use a shortened name for display
                    display_name = voice.name.split('-')[0].split('.')[0][:20]
                    VOICE_OPTIONS["pyttsx3"]["voices"][voice.id] = display_name
            except Exception as e:
                print(f"Error initializing pyttsx3: {e}")

        # Initialize ElevenLabs if API key is available
        if ELEVENLABS_AVAILABLE and self.config.get("elevenlabs_api_key"):
            try:
                if not ELEVENLABS_OLD_API:
                    from elevenlabs import ElevenLabs
                    self.elevenlabs_client = ElevenLabs(api_key=self.config.get("elevenlabs_api_key"))
                else:
                    set_api_key(self.config.get("elevenlabs_api_key"))
            except Exception as e:
                print(f"Error initializing ElevenLabs: {e}")

        # Initialize Azure if API key is available
        if AZURE_AVAILABLE and self.config.get("azure_tts_key"):
            try:
                self.azure_config = speechsdk.SpeechConfig(
                    subscription=self.config.get("azure_tts_key"),
                    region=self.config.get("azure_tts_region", "eastus")
                )
            except Exception as e:
                print(f"Error initializing Azure: {e}")

    def start_speech_processor(self):
        """Start a thread to process speech queue"""

        def processor():
            while True:
                try:
                    text, callback = self.speech_queue.get(timeout=0.1)
                    self.is_speaking = True
                    self._process_speech(text, callback)
                    self.is_speaking = False
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Speech processor error: {e}")
                    self.is_speaking = False

        thread = threading.Thread(target=processor, daemon=True)
        thread.start()

    def speak(self, text, callback=None):
        """Queue text for speech synthesis"""
        self.speech_queue.put((text, callback))

    def _process_speech(self, text, callback):
        """Actually convert text to speech and play it"""
        provider = self.config.get("voice_provider", "streamelements")
        voice = self.config.get("voice_name", "Brian")

        print(f"Speaking with {provider}/{voice}: {text[:50]}...")

        if provider == "streamelements":
            self._speak_streamelements(text, voice, callback)
        elif provider == "elevenlabs" and ELEVENLABS_AVAILABLE:
            self._speak_elevenlabs(text, voice, callback)
        elif provider == "azure" and AZURE_AVAILABLE:
            self._speak_azure(text, voice, callback)
        elif provider == "gtts" and GTTS_AVAILABLE:
            self._speak_gtts(text, voice, callback)
        elif provider == "pyttsx3" and PYTTSX3_AVAILABLE:
            self._speak_pyttsx3(text, voice, callback)
        else:
            # Fallback to StreamElements
            self._speak_streamelements(text, "Brian", callback)

    def _speak_streamelements(self, text, voice, callback):
        """Use StreamElements TTS"""
        audio_file = StreamElementsTTS.generate_speech(text, voice)
        if audio_file and PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error playing StreamElements audio: {e}")
            finally:
                try:
                    if audio_file:
                        os.remove(audio_file)
                except:
                    pass

        if callback:
            callback()

    def _speak_elevenlabs(self, text, voice, callback):
        """Use ElevenLabs API for speech"""
        if not self.config.get("elevenlabs_api_key"):
            self._speak_streamelements(text, "Brian", callback)
            return

        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.close()

            if not ELEVENLABS_OLD_API and self.elevenlabs_client:
                # New API
                audio = self.elevenlabs_client.generate(
                    text=text,
                    voice=voice,
                    model="eleven_monolingual_v1",
                    voice_settings=VoiceSettings(
                        stability=self.config.get("elevenlabs_stability", 0.5),
                        similarity_boost=self.config.get("elevenlabs_similarity", 0.75)
                    )
                )

                with open(temp_file.name, 'wb') as f:
                    for chunk in audio:
                        f.write(chunk)
            else:
                # Old API
                audio = generate(
                    text=text,
                    voice=voice,
                    model="eleven_monolingual_v1"
                )
                save(audio, temp_file.name)

            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                os.remove(temp_file.name)

            if callback:
                callback()
        except Exception as e:
            print(f"ElevenLabs error: {e}")
            self._speak_streamelements(text, "Brian", callback)

    def _speak_azure(self, text, voice, callback):
        """Use Azure TTS for speech"""
        if not self.azure_config:
            self._speak_streamelements(text, "Brian", callback)
            return

        try:
            # Set the voice
            self.azure_config.speech_synthesis_voice_name = voice

            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.azure_config)
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Play the audio if we have pygame
                if PYGAME_AVAILABLE:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_file.write(result.audio_data)
                    temp_file.close()

                    pygame.mixer.music.load(temp_file.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    os.remove(temp_file.name)

            if callback:
                callback()
        except Exception as e:
            print(f"Azure TTS error: {e}")
            self._speak_streamelements(text, "Brian", callback)

    def _speak_gtts(self, text, voice, callback):
        """Use Google TTS for speech"""
        if not GTTS_AVAILABLE:
            self._speak_streamelements(text, "Brian", callback)
            return

        try:
            from gtts import gTTS

            # voice is the language code
            tts = gTTS(text=text, lang=voice if voice != "en" else "en-us", slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            temp_file.close()

            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                os.remove(temp_file.name)

            if callback:
                callback()
        except Exception as e:
            print(f"Google TTS error: {e}")
            self._speak_streamelements(text, "Brian", callback)

    def _speak_pyttsx3(self, text, voice_id, callback):
        """Use pyttsx3 for local TTS"""
        if PYTTSX3_AVAILABLE and self.engine:
            try:
                # Set the voice if specified
                if voice_id and voice_id != "default":
                    self.engine.setProperty('voice', voice_id)

                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"pyttsx3 error: {e}")

        if callback:
            callback()


class EnhancedSpeechRecognition:
    """Enhanced speech recognition with simple recording"""

    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.is_recording = False
        self.recorded_audio = None
        self.recording_thread = None
        self.recording_start_time = None

        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            # Lower the energy threshold for better sensitivity
            self.recognizer.energy_threshold = 500  # Lower threshold
            self.recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment for consistency
            self.recognizer.pause_threshold = 1.5  # How long to wait for pause
            print(f"Speech recognizer initialized with energy threshold: {self.recognizer.energy_threshold}")

    def start_recording(self, device_index=None):
        """Start recording audio"""
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            print("Speech recognition not available")
            return False

        self.is_recording = True
        self.recorded_audio = None
        self.recording_start_time = time.time()

        def record():
            try:
                # Use specified device or default
                if device_index is not None and device_index >= 0:
                    print(f"Using microphone device index: {device_index}")
                    mic = sr.Microphone(device_index=device_index)
                else:
                    print("Using default microphone")
                    mic = sr.Microphone()

                with mic as source:
                    # Quick adjustment for ambient noise
                    print("Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Keep threshold low for sensitivity
                    if self.recognizer.energy_threshold > 1000:
                        self.recognizer.energy_threshold = 500
                    print(f"Energy threshold set to: {self.recognizer.energy_threshold}")

                    print("Listening for speech...")
                    # Keep listening while the button is held
                    audio_frames = []

                    # Use a loop to continuously capture audio while recording
                    while self.is_recording:
                        try:
                            # Listen for a short duration at a time
                            audio_chunk = self.recognizer.listen(source, timeout=0.1, phrase_time_limit=1)
                            audio_frames.append(audio_chunk)
                        except sr.WaitTimeoutError:
                            # No audio in this chunk, continue
                            pass

                        # Check if we've been recording for too long (safety limit)
                        if time.time() - self.recording_start_time > 30:
                            print("Recording time limit reached")
                            break

                    # Combine all audio frames
                    if audio_frames:
                        print(f"Captured {len(audio_frames)} audio chunks")
                        # Use the last continuous chunk for simplicity
                        # In production, you'd want to combine these properly
                        self.recorded_audio = audio_frames[-1] if len(audio_frames) == 1 else audio_frames[0]

                        # If we have multiple chunks, try to use the longest one
                        for chunk in audio_frames:
                            if hasattr(chunk, 'frame_data') and len(chunk.frame_data) > len(
                                    self.recorded_audio.frame_data):
                                self.recorded_audio = chunk
                    else:
                        print("No audio chunks captured")

            except Exception as e:
                print(f"Recording error: {e}")
                import traceback
                traceback.print_exc()
                self.recorded_audio = None

        self.recording_thread = threading.Thread(target=record, daemon=True)
        self.recording_thread.start()
        return True

    def stop_and_transcribe(self):
        """Stop recording and transcribe the audio"""
        self.is_recording = False

        # Wait for recording thread to finish (max 1 second)
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)

        if not self.recorded_audio:
            print("No audio was recorded")
            return None

        # Transcribe the recorded audio
        try:
            print("Transcribing audio with Google Speech Recognition...")
            # Try with different parameters
            try:
                text = self.recognizer.recognize_google(self.recorded_audio, language="en-US")
                print(f"Transcribed successfully: {text}")
                return text
            except sr.UnknownValueError:
                # Try with show_all to get alternatives
                print("Trying to get alternative transcriptions...")
                try:
                    result = self.recognizer.recognize_google(self.recorded_audio, language="en-US", show_all=True)
                    if result and 'alternative' in result:
                        text = result['alternative'][0].get('transcript', '')
                        if text:
                            print(f"Got alternative transcription: {text}")
                            return text
                except:
                    pass
                print("Google Speech Recognition could not understand the audio")
                return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None


class OBSImageOutput:
    """Manages image output for OBS integration"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.current_state = "idle"

    def update_obs_image(self, state="idle"):
        """Copy the appropriate image to the OBS output file"""
        if not self.config.get("obs_mode", True):
            return

        try:
            if state == "speaking":
                source_image = self.config.get("speaking_image", "speaking.png")
            else:
                source_image = self.config.get("idle_image", "idle.png")

            if os.path.exists(source_image):
                shutil.copy2(source_image, OBS_OUTPUT_FILE)
                self.current_state = state
            else:
                print(f"Image not found: {source_image}")
        except Exception as e:
            print(f"Error updating OBS image: {e}")


class AIManager:
    """Manages AI interactions with OpenAI"""

    def __init__(self, config_manager, conversation_history):
        self.config = config_manager
        self.conversation_history = conversation_history
        self.client = None
        self.chat_history = []
        self.init_ai()

    def init_ai(self):
        """Initialize OpenAI client"""
        api_key = self.config.get("openai_api_key")
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                self.chat_history = [{
                    "role": "system",
                    "content": self.config.get("system_prompt")
                }]

                # Load recent conversation history
                if self.config.get("save_conversation_history", True):
                    recent = self.conversation_history.get_recent(10)
                    for entry in recent:
                        if entry["role"] in ["user", "assistant"]:
                            self.chat_history.append({
                                "role": entry["role"],
                                "content": entry["content"]
                            })

                return True
            except Exception as e:
                print(f"OpenAI init error: {e}")
                return False
        return False

    def get_response(self, prompt, image_base64=None):
        """Get AI response with optional image"""
        if not self.client:
            self.init_ai()
            if not self.client:
                return "AI is not configured. Please set up OpenAI API key in settings."

        try:
            if image_base64:
                message_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            else:
                message_content = prompt

            self.chat_history.append({"role": "user", "content": message_content})

            # Save to conversation history
            if self.config.get("save_conversation_history", True):
                self.conversation_history.add_entry("user", prompt, {"has_image": bool(image_base64)})

            # Manage history length
            max_messages = self.config.get("ai_memory_messages", 20)
            while len(self.chat_history) > max_messages:
                if self.chat_history[1]["role"] != "system":
                    self.chat_history.pop(1)

            model_name = self.config.get("openai_model", "gpt-4o-mini")
            max_tokens = self.config.get("ai_max_tokens", 150)
            temperature = self.config.get("ai_temperature", 0.7)

            completion = self.client.chat.completions.create(
                model=model_name,
                messages=self.chat_history,
                max_tokens=max_tokens,
                temperature=temperature
            )

            response = completion.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": response})

            # Save response to conversation history
            if self.config.get("save_conversation_history", True):
                self.conversation_history.add_entry("assistant", response)

            return response
        except Exception as e:
            return f"Error getting AI response: {str(e)}"


class ScreenCapture:
    """Handles screen capture for AI vision"""

    @staticmethod
    def capture_screen_base64():
        """Capture screen and return as base64"""
        try:
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)

            height, width = screenshot_rgb.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                screenshot_rgb = cv2.resize(screenshot_rgb, (new_width, new_height))

            _, buffer = cv2.imencode('.jpg', screenshot_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None


class StyledButton(tk.Button):
    """Custom styled button with hover effects"""

    def __init__(self, parent, **kwargs):
        # Set default styling
        kwargs['bg'] = kwargs.get('bg', COLORS['button'])
        kwargs['fg'] = kwargs.get('fg', 'white')
        kwargs['font'] = kwargs.get('font', ('Comic Sans MS', 10, 'bold'))
        kwargs['relief'] = kwargs.get('relief', 'flat')
        kwargs['cursor'] = kwargs.get('cursor', 'hand2')
        kwargs['padx'] = kwargs.get('padx', 20)
        kwargs['pady'] = kwargs.get('pady', 10)
        kwargs['borderwidth'] = kwargs.get('borderwidth', 0)

        super().__init__(parent, **kwargs)

        # Add hover effects
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.default_bg = kwargs['bg']

    def on_enter(self, e):
        self['bg'] = COLORS['button_hover']

    def on_leave(self, e):
        self['bg'] = self.default_bg


class AIStreamerGUI:
    """Main GUI Application with beautiful lavender theme"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("💜 Lace's Simple TTS App 💜")
        self.root.geometry("1100x750")
        self.root.configure(bg=COLORS['bg'])

        # Check for required modules
        self.check_dependencies()

        # Set custom icon if available
        self.set_window_icon()

        # Set cute fonts
        self.setup_fonts()

        # Set style
        self.setup_styles()

        # Initialize managers
        self.config = ConfigManager()
        self.conversation_history = ConversationHistory()
        self.tts = TTSManager(self.config)
        self.ai = AIManager(self.config, self.conversation_history)
        self.obs_output = OBSImageOutput(self.config)
        self.speech_recognition = EnhancedSpeechRecognition()

        # Initialize Twitch with message callback
        self.twitch = TwitchChat(self.config, self.handle_twitch_message)

        # State variables
        self.is_speaking = False
        self.is_listening = False
        self.auto_commentary_active = False
        self.twitch_connected = False
        self.message_queue = queue.Queue()
        self.push_to_talk_held = False
        self.twitch_monitor_thread = None
        self.recording_thread = None
        self.voice_recording_active = False

        # Initialize config-based variables
        self.read_username_var = tk.BooleanVar(value=self.config.get("twitch_read_username", True))
        self.read_message_var = tk.BooleanVar(value=self.config.get("twitch_read_message", True))
        self.ai_response_var = tk.BooleanVar(value=self.config.get("twitch_ai_response", True))
        self.response_chance_var = tk.IntVar(value=int(self.config.get("twitch_response_chance", 0.8) * 100))

        # Create GUI
        self.create_gui()

        # Initialize OBS output
        self.obs_output.update_obs_image("idle")

        # Start processors
        self.process_messages()
        self.setup_push_to_talk()

        # Check for first run
        if not self.config.get("openai_api_key"):
            self.show_setup_wizard()

    def check_dependencies(self):
        """Check for required dependencies and show warnings"""
        missing = []

        if not PYAUDIO_AVAILABLE:
            missing.append("PyAudio (required for microphone)")
            print("WARNING: PyAudio not installed - microphone will not work!")
            print("Install with: pip install pyaudio")

        if not SPEECH_RECOGNITION_AVAILABLE:
            missing.append("SpeechRecognition (required for voice input)")
            print("WARNING: SpeechRecognition not installed - voice input will not work!")
            print("Install with: pip install SpeechRecognition")

        if not PYGAME_AVAILABLE:
            missing.append("Pygame (required for audio playback)")
            print("WARNING: Pygame not installed - audio playback may not work!")
            print("Install with: pip install pygame")

        if missing:
            print("\n" + "=" * 50)
            print("MISSING DEPENDENCIES:")
            for dep in missing:
                print(f"  • {dep}")
            print("=" * 50 + "\n")

    def set_window_icon(self):
        """Set custom window icon if available"""
        try:
            # Try multiple icon formats
            if os.path.exists(ICON_FILE):
                self.root.iconbitmap(ICON_FILE)
            elif os.path.exists("app_icon.png"):
                icon = Image.open("app_icon.png")
                photo = ImageTk.PhotoImage(icon)
                self.root.wm_iconphoto(True, photo)
            else:
                # Create a simple purple icon as fallback
                icon_img = Image.new('RGBA', (64, 64), COLORS['accent'])
                photo = ImageTk.PhotoImage(icon_img)
                self.root.wm_iconphoto(True, photo)
        except Exception as e:
            print(f"Could not set custom icon: {e}")

    def setup_fonts(self):
        """Setup cute fonts for the application"""
        # Try to use cute fonts, fallback to standard if not available
        try:
            self.title_font = font.Font(family="Comic Sans MS", size=20, weight="bold")
            self.header_font = font.Font(family="Comic Sans MS", size=14, weight="bold")
            self.normal_font = font.Font(family="Comic Sans MS", size=11)
            self.small_font = font.Font(family="Comic Sans MS", size=9)
        except:
            # Fallback fonts if Comic Sans not available
            self.title_font = font.Font(family="Arial Rounded MT Bold", size=20, weight="bold")
            self.header_font = font.Font(family="Arial", size=14, weight="bold")
            self.normal_font = font.Font(family="Arial", size=11)
            self.small_font = font.Font(family="Arial", size=9)

    def setup_styles(self):
        """Configure ttk styles for beautiful UI"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors for all widgets
        style.configure('TLabel', background=COLORS['frame_bg'], foreground=COLORS['text'],
                        font=('Comic Sans MS', 10))
        style.configure('Title.TLabel', font=('Comic Sans MS', 14, 'bold'), foreground=COLORS['accent'])
        style.configure('TFrame', background=COLORS['frame_bg'], relief='flat', borderwidth=2)
        style.configure('Card.TFrame', background=COLORS['frame_bg'], relief='ridge', borderwidth=2)
        style.configure('TLabelframe', background=COLORS['frame_bg'], foreground=COLORS['accent'])
        style.configure('TLabelframe.Label', background=COLORS['frame_bg'], foreground=COLORS['accent'],
                        font=('Comic Sans MS', 11, 'bold'))
        style.configure('TEntry', fieldbackground=COLORS['entry_bg'])
        style.configure('TCheckbutton', background=COLORS['frame_bg'], foreground=COLORS['text'],
                        font=('Comic Sans MS', 10), focuscolor='none')
        style.configure('TRadiobutton', background=COLORS['frame_bg'], foreground=COLORS['text'])
        style.configure('TSpinbox', fieldbackground=COLORS['entry_bg'])
        style.configure('TCombobox', fieldbackground=COLORS['entry_bg'])

        # Fix checkbox appearance to show checkmarks instead of X
        style.map('TCheckbutton',
                  background=[('active', COLORS['frame_bg']),
                              ('!disabled', COLORS['frame_bg'])],
                  indicatorcolor=[('selected', COLORS['success']),
                                  ('!selected', COLORS['entry_bg'])])

        # Notebook styling
        style.configure('TNotebook', background=COLORS['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', background=COLORS['frame_bg'], foreground=COLORS['text'],
                        padding=[20, 10], font=('Comic Sans MS', 10))
        style.map('TNotebook.Tab', background=[('selected', COLORS['accent'])],
                  foreground=[('selected', 'white')])

    def create_gui(self):
        """Create the main GUI layout"""
        # Title Header with cute styling
        header_frame = tk.Frame(self.root, bg=COLORS['accent'], height=70)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame, text="✨ Lace's Simple TTS App ✨",
                               font=self.title_font, bg=COLORS['accent'], fg='white')
        title_label.pack(expand=True)

        subtitle_label = tk.Label(header_frame, text="Your AI Streaming Companion 💜",
                                  font=self.small_font, bg=COLORS['accent'], fg='#F0E6FF')
        subtitle_label.pack()

        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Dashboard tab
        self.dashboard_frame = ttk.Frame(notebook)
        notebook.add(self.dashboard_frame, text="🏠 Dashboard")
        self.create_dashboard_tab()

        # Twitch tab
        self.twitch_frame = ttk.Frame(notebook)
        notebook.add(self.twitch_frame, text="💬 Twitch")
        self.create_twitch_tab()

        # Settings tab
        self.settings_frame = ttk.Frame(notebook)
        notebook.add(self.settings_frame, text="⚙️ Settings")
        self.create_settings_tab()

        # Chat/Log tab
        self.log_frame = ttk.Frame(notebook)
        notebook.add(self.log_frame, text="📝 Logs")
        self.create_log_tab()

    def create_dashboard_tab(self):
        """Create the main dashboard with cards layout"""
        # Main container with grid
        main_container = ttk.Frame(self.dashboard_frame)
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Left Column - Avatar and Status
        left_frame = ttk.Frame(main_container, style='Card.TFrame')
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        # Avatar Section
        avatar_label = ttk.Label(left_frame, text="AI Avatar", style='Title.TLabel')
        avatar_label.pack(pady=10)

        self.avatar_label = tk.Label(left_frame, text="No image loaded",
                                     bg='white', relief='sunken', font=self.normal_font)
        self.avatar_label.pack(fill='both', expand=True, padx=20, pady=10)

        # Status Section
        status_frame = ttk.LabelFrame(left_frame, text="Status")
        status_frame.pack(fill='x', padx=20, pady=10)

        self.status_indicators = {}
        statuses = [
            ("AI", "🤖", "Ready"),
            ("Voice", "🎤", "Inactive"),
            ("Twitch", "💬", "Disconnected"),
            ("OBS", "📹", "Active")
        ]

        for name, icon, default in statuses:
            frame = ttk.Frame(status_frame)
            frame.pack(fill='x', padx=10, pady=5)

            label = ttk.Label(frame, text=f"{icon} {name}:")
            label.pack(side='left')

            status_label = ttk.Label(frame, text=default, foreground=COLORS['success'])
            status_label.pack(side='right')
            self.status_indicators[name] = status_label

        # Right Column - Controls
        right_frame = ttk.Frame(main_container, style='Card.TFrame')
        right_frame.grid(row=0, column=1, sticky='nsew')

        controls_label = ttk.Label(right_frame, text="Quick Controls", style='Title.TLabel')
        controls_label.pack(pady=10)

        # Voice Control
        voice_frame = ttk.LabelFrame(right_frame, text="Voice Input")
        voice_frame.pack(fill='x', padx=20, pady=10)

        self.ptt_label = ttk.Label(voice_frame,
                                   text=f"Hold [{self.config.get('push_to_talk_key', 'V').upper()}] to talk")
        self.ptt_label.pack(pady=10)

        self.voice_indicator = tk.Canvas(voice_frame, width=200, height=30,
                                         bg=COLORS['frame_bg'], highlightthickness=0)
        self.voice_indicator.pack(pady=5)
        self.voice_level = self.voice_indicator.create_rectangle(0, 5, 0, 25,
                                                                 fill=COLORS['success'], outline='')

        # Auto Commentary
        auto_frame = ttk.LabelFrame(right_frame, text="Auto Commentary")
        auto_frame.pack(fill='x', padx=20, pady=10)

        self.auto_btn = StyledButton(auto_frame, text="Enable Auto Commentary",
                                     command=self.toggle_auto_commentary)
        self.auto_btn.pack(pady=10)

        # Interval control
        interval_frame = ttk.Frame(auto_frame)
        interval_frame.pack(pady=5)

        ttk.Label(interval_frame, text="Interval (seconds):").pack(side='left', padx=5)
        self.interval_spinbox = ttk.Spinbox(interval_frame, from_=5, to=300, width=10)
        self.interval_spinbox.pack(side='left', padx=5)
        self.interval_spinbox.set(self.config.get("commentary_interval", 15))

        # Save interval button
        StyledButton(interval_frame, text="Update",
                     command=self.update_commentary_interval,
                     font=self.small_font, padx=10, pady=5).pack(side='left', padx=5)

        self.auto_status = ttk.Label(auto_frame, text="Status: Disabled")
        self.auto_status.pack(pady=5)

        # Quick Actions
        actions_frame = ttk.LabelFrame(right_frame, text="Quick Actions")
        actions_frame.pack(fill='x', padx=20, pady=10)

        button_frame = ttk.Frame(actions_frame)
        button_frame.pack(pady=10)

        StyledButton(button_frame, text="📸 Test Screen",
                     command=self.test_screen_capture, width=15).pack(side='left', padx=5)
        StyledButton(button_frame, text="🔊 Test Voice",
                     command=self.test_speech, width=15).pack(side='left', padx=5)

        button_frame2 = ttk.Frame(actions_frame)
        button_frame2.pack(pady=5)

        StyledButton(button_frame2, text="🎤 Test Mic",
                     command=self.test_microphone, width=15).pack(side='left', padx=5)
        StyledButton(button_frame2, text="🗑️ Clear History",
                     command=self.clear_conversation_history, width=15,
                     bg=COLORS['error']).pack(side='left', padx=5)

        # Configure grid weights
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)

        # Load images
        self.load_avatar_images()

    def create_twitch_tab(self):
        """Create dedicated Twitch tab with connection and messages"""
        # Connection Section
        connection_frame = ttk.LabelFrame(self.twitch_frame, text="Twitch Connection")
        connection_frame.pack(fill='x', padx=20, pady=10)

        # Connection info
        info_frame = ttk.Frame(connection_frame)
        info_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(info_frame, text="📌 Quick Setup:", font=self.header_font).pack(anchor='w')
        ttk.Label(info_frame, text="1. Enter your Twitch username", font=self.small_font).pack(anchor='w', padx=20)
        ttk.Label(info_frame, text="2. Get OAuth from: https://twitchapps.com/tmi/",
                  font=self.small_font, foreground=COLORS['info']).pack(anchor='w', padx=20)
        ttk.Label(info_frame, text="3. Enter channel name (without #)", font=self.small_font).pack(anchor='w', padx=20)
        ttk.Label(info_frame, text="4. Click Connect!", font=self.small_font).pack(anchor='w', padx=20)

        # Connection controls
        control_frame = ttk.Frame(connection_frame)
        control_frame.pack(pady=10)

        self.twitch_connect_btn = StyledButton(control_frame, text="🔌 Connect to Twitch",
                                               command=self.toggle_twitch_connection,
                                               bg=COLORS['twitch'])
        self.twitch_connect_btn.pack(side='left', padx=5)

        self.twitch_connection_status = ttk.Label(control_frame, text="Not connected",
                                                  foreground=COLORS['text'])
        self.twitch_connection_status.pack(side='left', padx=20)

        # Twitch Messages Display
        messages_frame = ttk.LabelFrame(self.twitch_frame, text="Twitch Chat Messages")
        messages_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create twitch chat display
        self.twitch_chat_display = scrolledtext.ScrolledText(messages_frame, height=15, width=80,
                                                             bg='#1C1C2E', fg='white',
                                                             font=('Consolas', 10),
                                                             wrap=tk.WORD)
        self.twitch_chat_display.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure tags for different message types
        self.twitch_chat_display.tag_config('system', foreground='#FFD700', font=('Consolas', 10, 'italic'))
        self.twitch_chat_display.tag_config('username', foreground=COLORS['twitch'], font=('Consolas', 10, 'bold'))
        self.twitch_chat_display.tag_config('message', foreground='white')
        self.twitch_chat_display.tag_config('timestamp', foreground='#888888', font=('Consolas', 9))

        # Response Settings
        response_frame = ttk.LabelFrame(self.twitch_frame, text="TTS Settings for Twitch Chat")
        response_frame.pack(fill='x', padx=20, pady=10)

        settings_grid = ttk.Frame(response_frame)
        settings_grid.pack(padx=10, pady=10)

        # Reading options
        ttk.Label(settings_grid, text="📖 Reading Options:", font=self.header_font).grid(row=0, column=0, columnspan=2,
                                                                                        sticky='w', pady=(5, 10))

        ttk.Checkbutton(settings_grid, text="Read username before message",
                        variable=self.read_username_var).grid(row=1, column=0, columnspan=2, sticky='w', padx=20,
                                                              pady=2)

        ttk.Checkbutton(settings_grid, text="Read the message content",
                        variable=self.read_message_var).grid(row=2, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        ttk.Checkbutton(settings_grid, text="Generate AI response after reading",
                        variable=self.ai_response_var).grid(row=3, column=0, columnspan=2, sticky='w', padx=20, pady=2)

        # AI Response Settings (only applies when AI response is enabled)
        ttk.Label(settings_grid, text="🤖 AI Response Settings:", font=self.header_font).grid(row=4, column=0,
                                                                                             columnspan=2, sticky='w',
                                                                                             pady=(15, 10))

        ttk.Label(settings_grid, text="Response chance (%):", font=self.small_font).grid(row=5, column=0, sticky='w',
                                                                                         padx=20, pady=5)
        response_scale = ttk.Scale(settings_grid, from_=0, to=100, orient='horizontal',
                                   variable=self.response_chance_var, length=200)
        response_scale.grid(row=5, column=1, pady=5)
        self.response_chance_label = ttk.Label(settings_grid, text=f"{self.response_chance_var.get()}%")
        self.response_chance_label.grid(row=5, column=2, padx=10)

        # Update label when scale moves
        response_scale.configure(command=lambda v: self.response_chance_label.config(text=f"{int(float(v))}%"))

        ttk.Label(settings_grid, text="Cooldown (seconds):", font=self.small_font).grid(row=6, column=0, sticky='w',
                                                                                        padx=20, pady=5)
        self.cooldown_spinbox = ttk.Spinbox(settings_grid, from_=1, to=60, width=10)
        self.cooldown_spinbox.grid(row=6, column=1, sticky='w', pady=5)
        self.cooldown_spinbox.set(self.config.get("twitch_cooldown", 5))

    def create_settings_tab(self):
        """Create organized settings with cards"""
        # Create scrollable container
        canvas = tk.Canvas(self.settings_frame, bg=COLORS['frame_bg'])
        scrollbar = ttk.Scrollbar(self.settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Settings Grid Layout
        settings_container = ttk.Frame(scrollable_frame)
        settings_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Quick Setup Instructions Card (NEW)
        setup_card = ttk.LabelFrame(settings_container, text="📚 Quick Setup Guide")
        setup_card.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        setup_text = tk.Text(setup_card, height=8, width=80, wrap='word',
                             bg=COLORS['entry_bg'], fg=COLORS['text'], font=self.small_font)
        setup_text.pack(padx=10, pady=10)

        setup_instructions = """🤖 OpenAI Setup:
• Go to platform.openai.com/api-keys → Create API key → Paste below

💬 Twitch Setup:
• Username: Your Twitch username
• OAuth: Get from twitchapps.com/tmi/ (keep 'oauth:' prefix)
• Channel: Channel name to monitor (without #)

🎤 Voice: Hold V key to talk → Release to get AI response
📹 OBS: Add Image Source → Point to 'current_avatar.png' → Check 'Unload when not showing'"""

        setup_text.insert('1.0', setup_instructions)
        setup_text.config(state='disabled')

        # API Keys Card
        api_card = ttk.LabelFrame(settings_container, text="🔑 API Keys")
        api_card.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

        # OpenAI
        ttk.Label(api_card, text="OpenAI API Key:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.openai_entry = ttk.Entry(api_card, width=40, show='*')
        self.openai_entry.grid(row=0, column=1, padx=5, pady=5)
        self.openai_entry.insert(0, self.config.get("openai_api_key"))

        ttk.Label(api_card, text="Model:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.model_var = tk.StringVar(value=self.config.get("openai_model", "gpt-4o-mini"))
        model_combo = ttk.Combobox(api_card, textvariable=self.model_var, width=20,
                                   values=["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"])
        model_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)

        # ElevenLabs API
        ttk.Label(api_card, text="ElevenLabs API Key:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.elevenlabs_entry = ttk.Entry(api_card, width=40, show='*')
        self.elevenlabs_entry.grid(row=2, column=1, padx=5, pady=5)
        self.elevenlabs_entry.insert(0, self.config.get("elevenlabs_api_key", ""))

        # Azure API
        ttk.Label(api_card, text="Azure TTS Key:").grid(row=3, column=0, sticky='w', padx=10, pady=5)
        self.azure_entry = ttk.Entry(api_card, width=40, show='*')
        self.azure_entry.grid(row=3, column=1, padx=5, pady=5)
        self.azure_entry.insert(0, self.config.get("azure_tts_key", ""))

        ttk.Label(api_card, text="Azure Region:").grid(row=4, column=0, sticky='w', padx=10, pady=5)
        self.azure_region_entry = ttk.Entry(api_card, width=20)
        self.azure_region_entry.grid(row=4, column=1, sticky='w', padx=5, pady=5)
        self.azure_region_entry.insert(0, self.config.get("azure_tts_region", "eastus"))

        # Voice Settings Card (ENHANCED)
        voice_card = ttk.LabelFrame(settings_container, text="🎤 Voice Settings")
        voice_card.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

        ttk.Label(voice_card, text="Voice Provider:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.voice_provider_var = tk.StringVar(value=self.config.get("voice_provider", "streamelements"))
        self.voice_provider_combo = ttk.Combobox(voice_card, textvariable=self.voice_provider_var, width=20,
                                                 state='readonly')
        self.voice_provider_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(voice_card, text="Voice:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.voice_name_var = tk.StringVar(value=self.config.get("voice_name", "Brian"))
        self.voice_name_combo = ttk.Combobox(voice_card, textvariable=self.voice_name_var, width=20,
                                             state='readonly')
        self.voice_name_combo.grid(row=1, column=1, padx=5, pady=5)

        # Populate voice providers
        provider_names = []
        for provider_id, provider_info in VOICE_OPTIONS.items():
            provider_names.append(provider_id)
        self.voice_provider_combo['values'] = provider_names

        # Bind provider change event
        self.voice_provider_combo.bind('<<ComboboxSelected>>', self.on_voice_provider_changed)

        # Initialize voice names for current provider
        self.on_voice_provider_changed(None)

        ttk.Label(voice_card, text="Push-to-Talk Key:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.ptt_key_entry = ttk.Entry(voice_card, width=10)
        self.ptt_key_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        self.ptt_key_entry.insert(0, self.config.get("push_to_talk_key", "v"))

        # Test voice button
        StyledButton(voice_card, text="🔊 Test Voice",
                     command=self.test_current_voice,
                     font=self.small_font).grid(row=3, column=0, columnspan=2, pady=10)

        # Audio Device Settings Card
        audio_card = ttk.LabelFrame(settings_container, text="🔊 Audio Devices")
        audio_card.grid(row=2, column=0, sticky='ew', padx=5, pady=5)

        # Get available devices
        input_devices, output_devices = AudioDeviceManager.get_audio_devices()

        # Input device selection
        ttk.Label(audio_card, text="Microphone:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.input_device_var = tk.StringVar(value="Default")
        input_device_names = ["Default"] + [d['name'] for d in input_devices]
        input_combo = ttk.Combobox(audio_card, textvariable=self.input_device_var, width=30,
                                   values=input_device_names, state='readonly')
        input_combo.grid(row=0, column=1, padx=5, pady=5)

        # Set current selection
        current_input = self.config.get("input_device_index", -1)
        if current_input >= 0 and current_input < len(input_devices):
            input_combo.set(input_devices[current_input]['name'])

        # Output device selection
        ttk.Label(audio_card, text="Speakers:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.output_device_var = tk.StringVar(value="Default")
        output_device_names = ["Default"] + [d['name'] for d in output_devices]
        output_combo = ttk.Combobox(audio_card, textvariable=self.output_device_var, width=30,
                                    values=output_device_names, state='readonly')
        output_combo.grid(row=1, column=1, padx=5, pady=5)

        # Set current selection
        current_output = self.config.get("output_device_index", -1)
        if current_output >= 0 and current_output < len(output_devices):
            output_combo.set(output_devices[current_output]['name'])

        # Store device lists for saving
        self.input_devices = input_devices
        self.output_devices = output_devices

        # Twitch Settings Card
        twitch_card = ttk.LabelFrame(settings_container, text="💬 Twitch Settings")
        twitch_card.grid(row=3, column=0, sticky='ew', padx=5, pady=5)

        ttk.Label(twitch_card, text="Username:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.twitch_username_entry = ttk.Entry(twitch_card, width=30)
        self.twitch_username_entry.grid(row=0, column=1, padx=5, pady=5)
        self.twitch_username_entry.insert(0, self.config.get("twitch_username"))

        ttk.Label(twitch_card, text="OAuth Token:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.twitch_oauth_entry = ttk.Entry(twitch_card, width=30, show='*')
        self.twitch_oauth_entry.grid(row=1, column=1, padx=5, pady=5)
        self.twitch_oauth_entry.insert(0, self.config.get("twitch_oauth"))

        ttk.Label(twitch_card, text="Channel:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.twitch_channel_entry = ttk.Entry(twitch_card, width=30)
        self.twitch_channel_entry.grid(row=2, column=1, padx=5, pady=5)
        self.twitch_channel_entry.insert(0, self.config.get("twitch_channel"))

        # Avatar Settings Card
        avatar_card = ttk.LabelFrame(settings_container, text="🎭 Avatar Images")
        avatar_card.grid(row=2, column=1, rowspan=2, sticky='ew', padx=5, pady=5)

        ttk.Label(avatar_card, text="Idle Image:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.idle_path_label = ttk.Label(avatar_card, text=os.path.basename(self.config.get("idle_image", "")))
        self.idle_path_label.grid(row=0, column=1, padx=5, pady=5)
        StyledButton(avatar_card, text="Browse", command=lambda: self.browse_image("idle"),
                     bg=COLORS['info'], font=self.small_font).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(avatar_card, text="Speaking Image:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.speaking_path_label = ttk.Label(avatar_card, text=os.path.basename(self.config.get("speaking_image", "")))
        self.speaking_path_label.grid(row=1, column=1, padx=5, pady=5)
        StyledButton(avatar_card, text="Browse", command=lambda: self.browse_image("speaking"),
                     bg=COLORS['info'], font=self.small_font).grid(row=1, column=2, padx=5, pady=5)

        # AI Configuration Card (EXPANDED)
        ai_config_card = ttk.LabelFrame(settings_container, text="🧠 AI Configuration")
        ai_config_card.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        # AI settings grid
        ai_grid = ttk.Frame(ai_config_card)
        ai_grid.pack(padx=10, pady=10)

        # Memory settings
        ttk.Label(ai_grid, text="Message History:", font=self.header_font).grid(row=0, column=0, columnspan=2,
                                                                                sticky='w', pady=(0, 10))

        ttk.Label(ai_grid, text="Max messages to remember:").grid(row=1, column=0, sticky='w', padx=20, pady=5)
        self.memory_spinbox = ttk.Spinbox(ai_grid, from_=5, to=100, width=10)
        self.memory_spinbox.grid(row=1, column=1, sticky='w', pady=5)
        self.memory_spinbox.set(self.config.get("ai_memory_messages", 20))

        # Response settings
        ttk.Label(ai_grid, text="Response Settings:", font=self.header_font).grid(row=2, column=0, columnspan=2,
                                                                                  sticky='w', pady=(15, 10))

        ttk.Label(ai_grid, text="Max response length (tokens):").grid(row=3, column=0, sticky='w', padx=20, pady=5)
        self.tokens_spinbox = ttk.Spinbox(ai_grid, from_=50, to=500, width=10)
        self.tokens_spinbox.grid(row=3, column=1, sticky='w', pady=5)
        self.tokens_spinbox.set(self.config.get("ai_max_tokens", 150))

        ttk.Label(ai_grid, text="Creativity (0=focused, 1=creative):").grid(row=4, column=0, sticky='w', padx=20,
                                                                            pady=5)
        self.temperature_var = tk.DoubleVar(value=self.config.get("ai_temperature", 0.7))
        temp_scale = ttk.Scale(ai_grid, from_=0, to=1, orient='horizontal',
                               variable=self.temperature_var, length=150)
        temp_scale.grid(row=4, column=1, sticky='w', pady=5)
        self.temp_label = ttk.Label(ai_grid, text=f"{self.temperature_var.get():.1f}")
        self.temp_label.grid(row=4, column=2, padx=10)

        # Update temperature label
        temp_scale.configure(command=lambda v: self.temp_label.config(text=f"{float(v):.1f}"))

        # Conversation history option
        self.save_history_var = tk.BooleanVar(value=self.config.get("save_conversation_history", True))
        ttk.Checkbutton(ai_grid, text="Save conversation history to file",
                        variable=self.save_history_var).grid(row=5, column=0, columnspan=2, sticky='w', pady=10)

        # System Prompt Card
        prompt_card = ttk.LabelFrame(settings_container, text="💭 AI Personality & Instructions")
        prompt_card.grid(row=5, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        prompt_info = ttk.Label(prompt_card, text="Define how the AI should behave and respond:",
                                font=self.small_font, foreground=COLORS['info'])
        prompt_info.pack(padx=10, pady=(10, 5))

        self.prompt_text = scrolledtext.ScrolledText(prompt_card, height=6, width=60,
                                                     bg=COLORS['entry_bg'], fg=COLORS['text'],
                                                     font=self.small_font)
        self.prompt_text.pack(padx=10, pady=(0, 10))
        self.prompt_text.insert('1.0', self.config.get("system_prompt"))

        # Save Button
        save_frame = ttk.Frame(settings_container)
        save_frame.grid(row=6, column=0, columnspan=2, pady=20)

        StyledButton(save_frame, text="💾 Save All Settings",
                     command=self.save_settings, bg=COLORS['success']).pack()

        # Configure grid
        settings_container.columnconfigure(0, weight=1)
        settings_container.columnconfigure(1, weight=1)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_log_tab(self):
        """Create the chat/log tab with better styling"""
        # Chat display with custom colors
        chat_frame = ttk.LabelFrame(self.log_frame, text="System Logs")
        chat_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.chat_text = scrolledtext.ScrolledText(chat_frame, height=20, width=80,
                                                   bg='#2C3E50', fg='white',
                                                   font=('Consolas', 10),
                                                   insertbackground='white')
        self.chat_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Input frame
        input_frame = ttk.Frame(self.log_frame)
        input_frame.pack(fill='x', padx=20, pady=10)

        self.chat_entry = ttk.Entry(input_frame, font=self.normal_font)
        self.chat_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.chat_entry.bind('<Return>', lambda e: self.send_chat_message())

        StyledButton(input_frame, text="Send", command=self.send_chat_message).pack(side='left', padx=5)
        StyledButton(input_frame, text="Clear", command=self.clear_log,
                     bg=COLORS['error']).pack(side='left')

    def on_voice_provider_changed(self, event):
        """Handle voice provider change"""
        provider = self.voice_provider_var.get()

        if provider in VOICE_OPTIONS:
            voices = VOICE_OPTIONS[provider]["voices"]
            voice_names = list(voices.values())
            self.voice_name_combo['values'] = voice_names

            # Set to first voice or saved voice
            saved_voice = self.config.get("voice_name", "")
            if saved_voice in voice_names:
                self.voice_name_combo.set(saved_voice)
            elif voice_names:
                self.voice_name_combo.set(voice_names[0])

    def test_current_voice(self):
        """Test the currently selected voice"""
        provider = self.voice_provider_var.get()
        voice_display = self.voice_name_var.get()

        # Find the actual voice ID from display name
        voice_id = None
        if provider in VOICE_OPTIONS:
            for vid, vname in VOICE_OPTIONS[provider]["voices"].items():
                if vname == voice_display:
                    voice_id = vid
                    break

        if voice_id:
            # Temporarily update config for test
            old_provider = self.config.get("voice_provider")
            old_voice = self.config.get("voice_name")

            self.config.set("voice_provider", provider)
            self.config.set("voice_name", voice_id)

            test_text = f"Testing {voice_display} voice from {VOICE_OPTIONS[provider]['name']}. This is how I will sound!"
            self.message_queue.put(("ai", test_text))

            # Restore old config after a delay
            def restore():
                time.sleep(5)
                self.config.set("voice_provider", old_provider)
                self.config.set("voice_name", old_voice)

            threading.Thread(target=restore, daemon=True).start()

    def clear_conversation_history(self):
        """Clear the conversation history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all conversation history?"):
            self.conversation_history.clear()
            self.ai.chat_history = [{
                "role": "system",
                "content": self.config.get("system_prompt")
            }]
            messagebox.showinfo("Success", "Conversation history cleared!")

    def handle_twitch_message(self, msg_type, content):
        """Handle messages from Twitch connection"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if msg_type == "system":
            # System messages (connection status, etc.)
            self.twitch_chat_display.insert('end', f"[{timestamp}] ", 'timestamp')
            self.twitch_chat_display.insert('end', f"{content}\n", 'system')
        elif msg_type == "twitch_chat":
            # Chat messages from Twitch
            self.twitch_chat_display.insert('end', f"[{timestamp}] ", 'timestamp')

            # Parse username and message
            if ": " in content:
                username, message = content.split(": ", 1)
                self.twitch_chat_display.insert('end', f"{username}: ", 'username')
                self.twitch_chat_display.insert('end', f"{message}\n", 'message')

                # Add to main message queue for AI response
                self.message_queue.put(("twitch", content))

                # Process for TTS/AI response if connected
                if self.twitch_connected:
                    self.process_twitch_message_for_ai(username, message)
            else:
                self.twitch_chat_display.insert('end', f"{content}\n", 'message')

        self.twitch_chat_display.see('end')

    def process_twitch_message_for_ai(self, username, message):
        """Process Twitch message and generate AI response or just read it"""
        import random

        # Don't process if voice is active
        if self.voice_recording_active or self.is_listening:
            return

        # Always process if connected
        if not self.twitch_connected:
            return

        # Check if we're currently speaking - if so, skip this message
        if self.tts.is_speaking:
            return

        # Check cooldown (only for AI responses)
        current_time = time.time()
        cooldown = int(self.cooldown_spinbox.get()) if hasattr(self, 'cooldown_spinbox') else self.config.get(
            "twitch_cooldown", 5)

        if self.ai_response_var.get():
            # AI response mode - check cooldown and chance
            if current_time - self.twitch.last_response_time < cooldown:
                return

            # Check response chance
            chance = self.response_chance_var.get() / 100.0
            if random.random() > chance:
                return

            # Update last response time
            self.twitch.last_response_time = current_time

        # Process the message sequentially
        def process_sequential():
            # Step 1: Read username if enabled
            if self.read_username_var.get():
                username_text = f"{username} said"
                self.message_queue.put(("tts_only", username_text))

                # Wait for username to be spoken
                while self.tts.is_speaking:
                    time.sleep(0.1)

            # Step 2: Read message if enabled
            if self.read_message_var.get():
                self.message_queue.put(("tts_only", message))

                # Wait for message to be spoken
                while self.tts.is_speaking:
                    time.sleep(0.1)

            # Step 3: Generate and speak AI response if enabled
            if self.ai_response_var.get():
                # Small pause before AI response
                time.sleep(0.3)

                # Generate AI response
                response = self.ai.get_response(
                    f"Twitch viewer {username} said: {message}. Respond briefly and conversationally."
                )

                # Queue the AI response
                self.message_queue.put(("ai", response))

        # Run in a separate thread to not block
        thread = threading.Thread(target=process_sequential, daemon=True)
        thread.start()

    def toggle_twitch_connection(self):
        """Toggle Twitch chat connection"""
        if not self.twitch_connected:
            # Save current settings first
            self.config.set("twitch_username", self.twitch_username_entry.get())
            self.config.set("twitch_oauth", self.twitch_oauth_entry.get())
            self.config.set("twitch_channel", self.twitch_channel_entry.get())

            # Try to connect
            success, message = self.twitch.connect()

            if success:
                self.twitch_connected = True
                self.twitch_connect_btn.config(text="🔌 Disconnect")
                self.twitch_connection_status.config(text="✅ Connected!", foreground=COLORS['success'])
                self.update_status("Twitch", "Connected", COLORS['success'])
                self.start_twitch_monitor()
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Connection Failed", message)
        else:
            # Disconnect
            self.twitch.disconnect()
            self.twitch_connected = False
            self.twitch_connect_btn.config(text="🔌 Connect to Twitch")
            self.twitch_connection_status.config(text="Not connected", foreground=COLORS['text'])
            self.update_status("Twitch", "Disconnected", COLORS['text'])

    def start_twitch_monitor(self):
        """Monitor Twitch chat for messages"""

        def monitor_thread():
            while self.twitch_connected:
                try:
                    # Read messages from Twitch
                    messages = self.twitch.read_chat()
                    # Messages are already handled by the callback
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Twitch monitor error: {e}")
                    time.sleep(1)

        self.twitch_monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
        self.twitch_monitor_thread.start()

    def show_setup_wizard(self):
        """Show simplified setup instructions"""
        wizard = tk.Toplevel(self.root)
        wizard.title("Welcome to Lace's Simple TTS App!")
        wizard.geometry("700x500")
        wizard.configure(bg=COLORS['bg'])
        wizard.transient(self.root)

        # Header
        header = tk.Frame(wizard, bg=COLORS['accent'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(header, text="✨ Welcome! ✨",
                 font=self.title_font, bg=COLORS['accent'], fg='white').pack(expand=True)

        # Content
        content_frame = tk.Frame(wizard, bg=COLORS['frame_bg'])
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        tk.Label(content_frame, text="Let's get you started! 🚀",
                 font=self.header_font, bg=COLORS['frame_bg'], fg=COLORS['accent']).pack(pady=10)

        instructions_text = """The app is ready to use! Here's what you need to know:

1. 🤖 First Step: Get an OpenAI API Key
   • Go to the Settings tab
   • Enter your OpenAI API key
   • Click Save Settings

2. 🎤 Voice Control
   • Hold the V key to record your voice
   • Release to process and get AI response

3. 💬 Optional: Connect to Twitch
   • Enter your Twitch credentials in Settings
   • Go to Twitch tab and click Connect

4. 📹 For Streamers: OBS Setup
   • Add an Image Source in OBS
   • Point it to 'current_avatar.png'
   • The avatar will change when AI speaks!

All settings are in the Settings tab with detailed instructions.
Have fun streaming! 💜"""

        text_widget = tk.Text(content_frame, wrap='word', width=60, height=15,
                              bg=COLORS['entry_bg'], fg=COLORS['text'], font=self.normal_font,
                              padx=10, pady=10)
        text_widget.pack(pady=10)
        text_widget.insert('1.0', instructions_text)
        text_widget.config(state='disabled')

        # Close button
        StyledButton(content_frame, text="Got it! Let's go! 🚀",
                     command=wizard.destroy,
                     bg=COLORS['success']).pack(pady=20)

    def setup_push_to_talk(self):
        """Setup push-to-talk functionality with real speech recognition"""

        def ptt_listener():
            while True:
                try:
                    ptt_key = self.config.get("push_to_talk_key", "v")

                    # Check if key is pressed
                    if keyboard.is_pressed(ptt_key):
                        if not self.push_to_talk_held and not self.is_listening and not self.is_speaking:
                            # Stop any Twitch processing
                            self.voice_recording_active = True
                            self.push_to_talk_held = True
                            self.start_voice_recording()
                    else:
                        if self.push_to_talk_held:
                            self.push_to_talk_held = False
                            if self.is_listening:
                                self.stop_voice_recording()

                    time.sleep(0.05)  # Small delay to prevent high CPU usage
                except Exception as e:
                    print(f"PTT error: {e}")
                    time.sleep(0.1)

        thread = threading.Thread(target=ptt_listener, daemon=True)
        thread.start()

    def start_voice_recording(self):
        """Start recording voice with visual feedback"""
        if self.is_listening or self.is_speaking:
            return

        self.is_listening = True
        self.update_status("Voice", "🔴 Recording...", COLORS['error'])
        self.animate_voice_indicator(True)

        # Update UI
        if hasattr(self, 'ptt_label'):
            self.root.after(0, lambda: self.ptt_label.config(
                text=f"Recording... Release [{self.config.get('push_to_talk_key', 'V').upper()}] to stop",
                foreground=COLORS['error']))

        # Get device index
        device_index = self.config.get("input_device_index", -1)
        if device_index < 0:
            device_index = None

        # Start recording
        print(f"Starting voice recording with device index: {device_index}")
        success = self.speech_recognition.start_recording(device_index)
        if not success:
            self.message_queue.put(("system", "Failed to start recording. Check your microphone."))
            self.is_listening = False
            self.update_status("Voice", "Error", COLORS['error'])

    def stop_voice_recording(self):
        """Stop recording and process the audio"""
        if not self.is_listening:
            return

        self.is_listening = False
        self.update_status("Voice", "Processing...", COLORS['info'])
        self.animate_voice_indicator(False)

        # Update UI
        if hasattr(self, 'ptt_label'):
            self.root.after(0, lambda: self.ptt_label.config(
                text=f"Hold [{self.config.get('push_to_talk_key', 'V').upper()}] to talk",
                foreground=COLORS['text']))

        # Process recording in thread
        def process_thread():
            try:
                print("Stopping recording and transcribing...")
                # Stop recording and get transcription
                text = self.speech_recognition.stop_and_transcribe()

                if text and text.strip():
                    print(f"Successfully transcribed: {text}")
                    # Show what was heard
                    self.message_queue.put(("user", f"[Voice]: {text}"))

                    # Get screen capture
                    screen_b64 = ScreenCapture.capture_screen_base64()

                    # Get AI response
                    prompt = f"User said: {text}"
                    if screen_b64:
                        prompt += " (Also consider what's visible on screen)"

                    response = self.ai.get_response(prompt, screen_b64)
                    self.message_queue.put(("ai", response))
                else:
                    print("No speech detected or transcription failed")
                    self.message_queue.put(("system",
                                            "No speech detected. Make sure:\n• Your microphone is working\n• You're speaking clearly\n• The correct microphone is selected in Settings"))

                # Update status
                self.update_status("Voice", "Ready", COLORS['success'])

            except Exception as e:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()
                self.message_queue.put(("system", f"Voice processing error: {str(e)}"))
                self.update_status("Voice", "Error", COLORS['error'])
            finally:
                # Re-enable other processing
                self.voice_recording_active = False

        self.recording_thread = threading.Thread(target=process_thread, daemon=True)
        self.recording_thread.start()

    def animate_voice_indicator(self, recording):
        """Animate the voice level indicator"""
        if recording:
            # Show that we're recording with a pulsing effect
            def animate():
                if self.is_listening:
                    # Create a pulsing effect to show recording is active
                    import math
                    import time
                    pulse = int(100 + 80 * math.sin(time.time() * 3))
                    self.voice_indicator.coords(self.voice_level, 0, 5, pulse, 25)
                    self.voice_indicator.itemconfig(self.voice_level, fill=COLORS['error'])
                    self.root.after(50, animate)

            animate()
        else:
            # Reset indicator
            self.voice_indicator.coords(self.voice_level, 0, 5, 0, 25)
            self.voice_indicator.itemconfig(self.voice_level, fill=COLORS['success'])

    def update_commentary_interval(self):
        """Update the auto commentary interval"""
        try:
            interval = int(self.interval_spinbox.get())
            self.config.set("commentary_interval", interval)
            self.auto_status.config(
                text=f"Status: {'Active' if self.auto_commentary_active else 'Disabled'} ({interval}s)")
            messagebox.showinfo("Updated", f"Commentary interval set to {interval} seconds")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")

    def toggle_auto_commentary(self):
        """Toggle automatic commentary"""
        self.auto_commentary_active = not self.auto_commentary_active

        if self.auto_commentary_active:
            self.auto_btn.config(text="Disable Auto Commentary")
            self.auto_status.config(text="Status: Active", foreground=COLORS['success'])
            self.start_auto_commentary()
        else:
            self.auto_btn.config(text="Enable Auto Commentary")
            self.auto_status.config(text="Status: Disabled", foreground=COLORS['text'])

    def start_auto_commentary(self):
        """Start automatic commentary thread"""

        def commentary_thread():
            interval = self.config.get("commentary_interval", 15)
            while self.auto_commentary_active:
                waited = 0
                while waited < interval and self.auto_commentary_active:
                    time.sleep(0.5)
                    waited += 0.5

                if not self.auto_commentary_active:
                    break

                # Don't comment if voice is active
                if not self.is_speaking and not self.is_listening and not self.voice_recording_active:
                    screen_b64 = ScreenCapture.capture_screen_base64()
                    if screen_b64:
                        response = self.ai.get_response(
                            "Comment briefly on what you see on screen (1-2 sentences max)",
                            screen_b64
                        )
                        self.message_queue.put(("ai", response))

        thread = threading.Thread(target=commentary_thread, daemon=True)
        thread.start()

    def update_status(self, name, status, color=None):
        """Update status indicator"""
        if name in self.status_indicators:
            self.status_indicators[name].config(text=status)
            if color:
                self.status_indicators[name].config(foreground=color)

    def test_speech(self):
        """Test speech synthesis"""
        test_text = "Hello! Testing Lace's Simple TTS App! The voice system is working perfectly!"
        self.message_queue.put(("ai", test_text))

    def test_microphone(self):
        """Test microphone functionality"""
        self.message_queue.put(("system", "Testing microphone... Speak now!"))

        def test_thread():
            try:
                # Test basic microphone access
                if not SPEECH_RECOGNITION_AVAILABLE:
                    self.message_queue.put(("system", "❌ Speech recognition module not installed!"))
                    return

                if not PYAUDIO_AVAILABLE:
                    self.message_queue.put(
                        ("system", "❌ PyAudio not installed! Please install it for microphone support."))
                    return

                # Get device index
                device_index = self.config.get("input_device_index", -1)
                if device_index < 0:
                    device_index = None

                # Try to access microphone
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 1000

                try:
                    if device_index is not None:
                        mic = sr.Microphone(device_index=device_index)
                        self.message_queue.put(("system", f"Using microphone device {device_index}"))
                    else:
                        mic = sr.Microphone()
                        self.message_queue.put(("system", "Using default microphone"))

                    with mic as source:
                        self.message_queue.put(("system", "Adjusting for ambient noise..."))
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        self.message_queue.put(("system", f"Energy threshold: {recognizer.energy_threshold}"))
                        self.message_queue.put(("system", "Listening for 5 seconds... Say something!"))

                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                        self.message_queue.put(("system", "Processing audio..."))
                        text = recognizer.recognize_google(audio)
                        self.message_queue.put(("system", f"✅ Microphone works! Heard: '{text}'"))

                except sr.WaitTimeoutError:
                    self.message_queue.put(
                        ("system", "⚠️ No speech detected. Microphone may be working but too quiet."))
                except sr.UnknownValueError:
                    self.message_queue.put(("system", "⚠️ Microphone detected sound but couldn't understand speech."))
                except sr.RequestError as e:
                    self.message_queue.put(("system", f"❌ Google Speech API error: {e}"))
                except OSError as e:
                    if "Invalid number of channels" in str(e):
                        self.message_queue.put(("system",
                                                "❌ Microphone configuration error. Try selecting a different device in Settings."))
                    else:
                        self.message_queue.put(("system", f"❌ Microphone access error: {e}"))
                except Exception as e:
                    self.message_queue.put(("system", f"❌ Microphone error: {e}"))

            except Exception as e:
                self.message_queue.put(("system", f"❌ Test failed: {e}"))
                import traceback
                print(traceback.format_exc())

        thread = threading.Thread(target=test_thread, daemon=True)
        thread.start()

    def test_screen_capture(self):
        """Test screen capture"""
        screen_b64 = ScreenCapture.capture_screen_base64()
        if screen_b64:
            response = self.ai.get_response("Describe what you see on the screen briefly.", screen_b64)
            self.message_queue.put(("ai", response))
        else:
            messagebox.showwarning("Screen Capture", "Failed to capture screen")

    def browse_image(self, image_type):
        """Browse for an image file"""
        filename = filedialog.askopenfilename(
            title=f"Select {image_type} image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if filename:
            if image_type == "idle":
                self.config.set("idle_image", filename)
                self.idle_path_label.config(text=os.path.basename(filename))
            else:
                self.config.set("speaking_image", filename)
                self.speaking_path_label.config(text=os.path.basename(filename))
            self.load_avatar_images()
            self.obs_output.update_obs_image("idle")

    def load_avatar_images(self):
        """Load and prepare avatar images"""
        try:
            idle_path = self.config.get("idle_image")
            if idle_path and os.path.exists(idle_path):
                idle_img = Image.open(idle_path)
                idle_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                self.idle_photo = ImageTk.PhotoImage(idle_img)
            else:
                self.idle_photo = None

            speaking_path = self.config.get("speaking_image")
            if speaking_path and os.path.exists(speaking_path):
                speaking_img = Image.open(speaking_path)
                speaking_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                self.speaking_photo = ImageTk.PhotoImage(speaking_img)
            else:
                self.speaking_photo = None

            self.show_idle_image()
        except Exception as e:
            print(f"Error loading images: {e}")

    def show_idle_image(self):
        """Display the idle image"""
        if self.idle_photo:
            self.avatar_label.config(image=self.idle_photo, text="")
        else:
            self.avatar_label.config(image="", text="No idle image", bg='white')
        self.is_speaking = False
        self.obs_output.update_obs_image("idle")

    def show_speaking_image(self):
        """Display the speaking image"""
        if self.speaking_photo:
            self.avatar_label.config(image=self.speaking_photo, text="")
        else:
            self.avatar_label.config(image="", text="No speaking image", bg='white')
        self.is_speaking = True
        self.obs_output.update_obs_image("speaking")

    def save_settings(self):
        """Save all settings"""
        # Save API keys
        self.config.set("openai_api_key", self.openai_entry.get())
        self.config.set("openai_model", self.model_var.get())
        self.config.set("elevenlabs_api_key", self.elevenlabs_entry.get())
        self.config.set("azure_tts_key", self.azure_entry.get())
        self.config.set("azure_tts_region", self.azure_region_entry.get())

        # Save voice settings
        provider = self.voice_provider_var.get()
        voice_display = self.voice_name_var.get()

        # Find the actual voice ID from display name
        voice_id = voice_display  # Default to display name
        if provider in VOICE_OPTIONS:
            for vid, vname in VOICE_OPTIONS[provider]["voices"].items():
                if vname == voice_display:
                    voice_id = vid
                    break

        self.config.set("voice_provider", provider)
        self.config.set("voice_name", voice_id)
        self.config.set("push_to_talk_key", self.ptt_key_entry.get())

        # Save Twitch settings
        self.config.set("twitch_username", self.twitch_username_entry.get())
        self.config.set("twitch_oauth", self.twitch_oauth_entry.get())
        self.config.set("twitch_channel", self.twitch_channel_entry.get())
        self.config.set("twitch_read_username", self.read_username_var.get())
        self.config.set("twitch_read_message", self.read_message_var.get())
        self.config.set("twitch_ai_response", self.ai_response_var.get())
        self.config.set("twitch_response_chance", self.response_chance_var.get() / 100.0)
        self.config.set("twitch_cooldown", int(self.cooldown_spinbox.get()))
        self.config.set("commentary_interval",
                        int(self.interval_spinbox.get()) if hasattr(self, 'interval_spinbox') else 15)
        self.config.set("system_prompt", self.prompt_text.get('1.0', 'end-1c'))

        # Save AI configuration settings
        self.config.set("ai_memory_messages", int(self.memory_spinbox.get()))
        self.config.set("ai_max_tokens", int(self.tokens_spinbox.get()))
        self.config.set("ai_temperature", float(self.temperature_var.get()))
        self.config.set("save_conversation_history", self.save_history_var.get())

        # Save audio device selections
        if hasattr(self, 'input_devices'):
            selected_input = self.input_device_var.get()
            if selected_input == "Default":
                self.config.set("input_device_index", -1)
            else:
                for device in self.input_devices:
                    if device['name'] == selected_input:
                        self.config.set("input_device_index", device['index'])
                        break

        if hasattr(self, 'output_devices'):
            selected_output = self.output_device_var.get()
            if selected_output == "Default":
                self.config.set("output_device_index", -1)
            else:
                for device in self.output_devices:
                    if device['name'] == selected_output:
                        self.config.set("output_device_index", device['index'])
                        AudioDeviceManager.set_pygame_output_device(device['index'])
                        break

        if hasattr(self, 'ptt_label'):
            self.ptt_label.config(text=f"Hold [{self.ptt_key_entry.get().upper()}] to talk")

        # Re-initialize managers with new settings
        self.tts.init_tts()
        self.ai.init_ai()

        messagebox.showinfo("Success", "Settings saved successfully! 💜")

    def send_chat_message(self):
        """Send a chat message to the AI"""
        message = self.chat_entry.get().strip()
        if message:
            self.chat_entry.delete(0, 'end')
            self.message_queue.put(("user", message))

            response = self.ai.get_response(message)
            self.message_queue.put(("ai", response))

    def clear_log(self):
        """Clear the chat log"""
        self.chat_text.delete('1.0', tk.END)

    def process_messages(self):
        """Process messages from the queue"""
        try:
            while True:
                sender, message = self.message_queue.get_nowait()

                timestamp = time.strftime("%H:%M:%S")

                if sender == "ai":
                    self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
                    self.chat_text.insert('end', f"AI: ", 'ai')
                    self.chat_text.insert('end', f"{message}\n", 'ai_message')

                    self.show_speaking_image()
                    self.tts.speak(message, callback=self.show_idle_image)

                elif sender == "tts_only":
                    # Just speak without logging as AI response
                    self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
                    self.chat_text.insert('end', f"TTS: ", 'system')
                    self.chat_text.insert('end', f"{message}\n", 'system_message')

                    self.show_speaking_image()
                    self.tts.speak(message, callback=self.show_idle_image)

                elif sender == "user":
                    self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
                    self.chat_text.insert('end', f"YOU: ", 'user')
                    self.chat_text.insert('end', f"{message}\n", 'user_message')

                elif sender == "twitch":
                    self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
                    self.chat_text.insert('end', f"TWITCH: ", 'twitch')
                    self.chat_text.insert('end', f"{message}\n", 'twitch_message')

                elif sender == "system":
                    self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
                    self.chat_text.insert('end', f"SYSTEM: ", 'system')
                    self.chat_text.insert('end', f"{message}\n", 'system_message')

                # Configure text tags
                self.chat_text.tag_config('timestamp', foreground='#7F8C8D')
                self.chat_text.tag_config('ai', foreground='#E74C3C', font=('Arial', 10, 'bold'))
                self.chat_text.tag_config('ai_message', foreground='#ECF0F1')
                self.chat_text.tag_config('user', foreground='#3498DB', font=('Arial', 10, 'bold'))
                self.chat_text.tag_config('user_message', foreground='#ECF0F1')
                self.chat_text.tag_config('twitch', foreground='#9B59B6', font=('Arial', 10, 'bold'))
                self.chat_text.tag_config('twitch_message', foreground='#ECF0F1')
                self.chat_text.tag_config('system', foreground='#F39C12', font=('Arial', 10, 'bold'))
                self.chat_text.tag_config('system_message', foreground='#BDC3C7')

                self.chat_text.see('end')
        except queue.Empty:
            pass

        self.root.after(100, self.process_messages)

    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = AIStreamerGUI()
    app.run()
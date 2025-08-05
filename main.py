import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
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

# Optional imports (will check if available)
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from elevenlabs import generate, save, set_api_key, voices

    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

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

# Constants
CONFIG_FILE = "config.json"
OBS_OUTPUT_FILE = "current_avatar.png"  # File that OBS will monitor
DEFAULT_CONFIG = {
    "openai_api_key": "",
    "elevenlabs_api_key": "",
    "azure_tts_key": "",
    "azure_tts_region": "eastus",
    "twitch_username": "",
    "twitch_oauth": "",
    "twitch_channel": "",
    "ai_voice": "streamelements_brian",  # Defaults the voice to StreamElements's "BRIAN" voice
    "elevenlabs_voice_name": "",
    "idle_image": "idle.png",
    "speaking_image": "speaking.png",
    "system_prompt": "You are a helpful, albeit sassy individual watching a stream and interacting with the streamer and chat.",
    "voice_key": "v",
    "stop_voice_key": "p",
    "auto_commentary": False,
    "commentary_interval": 15,
    "respond_to_chat": True,
    "obs_mode": True,  # Enable OBS output by default
    "transparent_window": False,  # Option for transparent overlay window
    # Selectable OpenAI model (defaults to GPT-4o mini for cost efficiency)
    "openai_model": "gpt-4o-mini"
}


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
                    # Merge with defaults to ensure all keys exist
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


class StreamElementsTTS:
    """StreamElements TTS API integration for Brian voice"""

    @staticmethod
    def generate_speech(text, voice="Brian"):
        """Generate speech using StreamElements API"""
        url = "https://api.streamelements.com/kappa/v2/speech"
        params = {
            "voice": voice,
            "text": text
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                # Save to temporary file
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
        self.init_tts()

    def init_tts(self):
        """Initialize TTS engine based on availability"""
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=48000, buffer=1024)

        if PYTTSX3_AVAILABLE:
            self.engine = pyttsx3.init()
            # Set properties for backup voice
            self.engine.setProperty('rate', 150)
            voices = self.engine.getProperty('voices')
            # Try to find a male voice
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

    def speak(self, text, callback=None):
        """Convert text to speech and play it"""
        voice_type = self.config.get("ai_voice", "streamelements_brian")

        if voice_type == "streamelements_brian":
            self._speak_streamelements(text, callback)
        elif voice_type == "elevenlabs" and ELEVENLABS_AVAILABLE:
            self._speak_elevenlabs(text, callback)
        elif voice_type == "azure" and AZURE_AVAILABLE:
            self._speak_azure(text, callback)
        else:
            # Fallback to pyttsx3
            self._speak_pyttsx3(text, callback)

    def _speak_streamelements(self, text, callback):
        """Use StreamElements Brian voice for TTS.

        This implementation isolates playback and cleanup so that only genuine
        playback errors trigger a fallback to pyttsx3. Any issues deleting
        temporary files will not cause a second voice to play. The provided
        callback is invoked exactly once after the audio completes (or after
        fallback), ensuring consistent behavior.
        """

        def speak_thread():
            audio_file = StreamElementsTTS.generate_speech(text, "Brian")
            # If an audio file was generated and pygame is available, attempt playback
            if audio_file and PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    # If playback fails, fall back to pyttsx3 and exit early
                    print(f"Error playing StreamElements audio: {e}")
                    self._speak_pyttsx3(text, callback)
                    return
                finally:
                    # Attempt to remove the temporary audio file; ignore errors
                    if audio_file:
                        try:
                            os.remove(audio_file)
                        except Exception:
                            pass
                # Invoke callback after successful playback
                if callback:
                    callback()
            else:
                # No audio file or no pygame: fall back to pyttsx3 immediately
                self._speak_pyttsx3(text, callback)

        # Run playback in a daemon thread to avoid blocking the GUI
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()

    def _speak_pyttsx3(self, text, callback):
        """Use pyttsx3 for local TTS"""
        if PYTTSX3_AVAILABLE and self.engine:
            def speak_thread():
                self.engine.say(text)
                self.engine.runAndWait()
                if callback:
                    callback()

            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
        elif callback:
            callback()

    def _speak_elevenlabs(self, text, callback):
        """Use ElevenLabs API for speech"""
        if ELEVENLABS_AVAILABLE:
            try:
                set_api_key(self.config.get("elevenlabs_api_key"))
                voice_name = self.config.get("elevenlabs_voice_name", "Antoni")

                def speak_thread():
                    audio = generate(text=text, voice=voice_name, model="eleven_monolingual_v1")
                    # Save to temporary file
                    temp_file = "temp_audio.mp3"
                    save(audio, temp_file)

                    # Play with pygame
                    if PYGAME_AVAILABLE:
                        pygame.mixer.music.load(temp_file)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        os.remove(temp_file)

                    if callback:
                        callback()

                thread = threading.Thread(target=speak_thread)
                thread.daemon = True
                thread.start()
            except Exception as e:
                print(f"ElevenLabs error: {e}")
                self._speak_streamelements(text, callback)  # Fallback
        else:
            self._speak_streamelements(text, callback)

    def _speak_azure(self, text, callback):
        """Use Azure TTS for speech"""
        if AZURE_AVAILABLE:
            try:
                speech_config = speechsdk.SpeechConfig(
                    subscription=self.config.get("azure_tts_key"),
                    region=self.config.get("azure_tts_region")
                )
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

                def speak_thread():
                    synthesizer.speak_text_async(text).get()
                    if callback:
                        callback()

                thread = threading.Thread(target=speak_thread)
                thread.daemon = True
                thread.start()
            except Exception as e:
                print(f"Azure TTS error: {e}")
                self._speak_streamelements(text, callback)  # Fallback
        else:
            self._speak_streamelements(text, callback)


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
                # Copy image to OBS output file
                shutil.copy2(source_image, OBS_OUTPUT_FILE)
                self.current_state = state
            else:
                print(f"Image not found: {source_image}")
        except Exception as e:
            print(f"Error updating OBS image: {e}")


class AIManager:
    """Manages AI interactions with OpenAI"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.client = None
        self.chat_history = []
        self.init_ai()

    def init_ai(self):
        """Initialize OpenAI client"""
        api_key = self.config.get("openai_api_key")
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                # Initialize with system prompt
                self.chat_history = [{
                    "role": "system",
                    "content": self.config.get("system_prompt")
                }]
                return True
            except Exception as e:
                print(f"OpenAI init error: {e}")
                return False
        return False

    def get_response(self, prompt, image_base64=None):
        """Get AI response with optional image"""
        if not self.client:
            # Reinitialize if needed
            self.init_ai()
            if not self.client:
                return "AI is not configured. Please set up OpenAI API key in settings."

        try:
            # Prepare message
            if image_base64:
                message_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            else:
                message_content = prompt

            # Add to history
            self.chat_history.append({"role": "user", "content": message_content})

            # Trim history if too long (keep system message)
            while len(self.chat_history) > 20:
                if self.chat_history[1]["role"] != "system":
                    self.chat_history.pop(1)

            # Get response
            # Use the model specified in configuration, defaulting to GPT-4o mini
            model_name = self.config.get("openai_model", "gpt-4o-mini")
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=self.chat_history,
                max_tokens=150  # Keep responses concise
            )

            response = completion.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": response})

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

            # Resize for efficiency
            height, width = screenshot_rgb.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                screenshot_rgb = cv2.resize(screenshot_rgb, (new_width, new_height))

            # Encode to base64
            _, buffer = cv2.imencode('.jpg', screenshot_rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None


class SpeechToText:
    """Handles speech recognition"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.recognizer = None
        self.init_stt()

    def init_stt(self):
        """Initialize speech recognition"""
        if AZURE_AVAILABLE and self.config.get("azure_tts_key"):
            try:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=self.config.get("azure_tts_key"),
                    region=self.config.get("azure_tts_region")
                )
                self.speech_config.speech_recognition_language = "en-US"
                return True
            except:
                return False
        return False

    def listen(self, stop_key='p'):
        """Listen to microphone until stop key is pressed"""
        if not AZURE_AVAILABLE:
            return "Speech recognition not available. Please install Azure Speech SDK or type your message in the chat."

        try:
            recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config)

            all_results = []
            done = False

            def handle_result(evt):
                all_results.append(evt.result.text)

            def stop_cb(evt):
                nonlocal done
                done = True

            recognizer.recognized.connect(handle_result)
            recognizer.session_stopped.connect(stop_cb)
            recognizer.canceled.connect(stop_cb)

            recognizer.start_continuous_recognition_async().get()

            # Wait for stop key
            print(f"Listening... Press '{stop_key}' to stop")
            while not done:
                if keyboard.is_pressed(stop_key):
                    recognizer.stop_continuous_recognition_async()
                    break
                time.sleep(0.1)

            return " ".join(all_results).strip()
        except Exception as e:
            return f"Error: {str(e)}"


class AIStreamerGUI:
    """Main GUI Application"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Streamer Assistant")
        self.root.geometry("900x700")

        # Initialize managers
        self.config = ConfigManager()
        self.tts = TTSManager(self.config)
        self.ai = AIManager(self.config)
        self.stt = SpeechToText(self.config)
        self.obs_output = OBSImageOutput(self.config)

        # State variables
        self.is_speaking = False
        self.is_listening = False
        self.auto_commentary_active = False
        self.message_queue = queue.Queue()

        # Image paths
        self.idle_image_path = None
        self.speaking_image_path = None
        self.current_image_label = None

        # Create GUI
        self.create_gui()

        # Initialize OBS output
        self.obs_output.update_obs_image("idle")

        # Start message processor
        self.process_messages()

        # Check for first run
        if not self.config.get("openai_api_key"):
            self.show_setup_wizard()

    def create_gui(self):
        """Create the main GUI layout"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Main tab
        self.main_frame = ttk.Frame(notebook)
        notebook.add(self.main_frame, text="Main")
        self.create_main_tab()

        # Settings tab
        self.settings_frame = ttk.Frame(notebook)
        notebook.add(self.settings_frame, text="Settings")
        self.create_settings_tab()

        # Chat/Log tab
        self.log_frame = ttk.Frame(notebook)
        notebook.add(self.log_frame, text="Chat/Logs")
        self.create_log_tab()

    def create_main_tab(self):
        """Create the main control tab"""
        # Avatar display
        avatar_frame = ttk.LabelFrame(self.main_frame, text="AI Avatar")
        avatar_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.avatar_label = tk.Label(avatar_frame, text="No image loaded", bg='gray')
        self.avatar_label.pack(fill='both', expand=True, padx=10, pady=10)

        # OBS Integration info
        obs_frame = ttk.LabelFrame(self.main_frame, text="OBS Integration")
        obs_frame.pack(fill='x', padx=10, pady=5)

        obs_info = ttk.Label(obs_frame,
                             text=f"Add 'Image Source' in OBS → Browse to: {os.path.abspath(OBS_OUTPUT_FILE)}")
        obs_info.pack(padx=10, pady=5)

        self.obs_mode_var = tk.BooleanVar(value=self.config.get("obs_mode", True))
        obs_checkbox = ttk.Checkbutton(obs_frame, text="Enable OBS Output", variable=self.obs_mode_var,
                                       command=self.toggle_obs_mode)
        obs_checkbox.pack(padx=10, pady=5)

        # Control buttons
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill='x', padx=10, pady=5)

        self.voice_btn = ttk.Button(control_frame, text="Start Voice Input (V)", command=self.toggle_voice_input)
        self.voice_btn.pack(side='left', padx=5)

        self.auto_btn = ttk.Button(control_frame, text="Toggle Auto Commentary", command=self.toggle_auto_commentary)
        self.auto_btn.pack(side='left', padx=5)

        ttk.Button(control_frame, text="Test Speech", command=self.test_speech).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Capture Screen", command=self.test_screen_capture).pack(side='left', padx=5)

        # Status bar
        self.status_label = ttk.Label(self.main_frame, text="Ready", relief='sunken')
        self.status_label.pack(fill='x', side='bottom', padx=10, pady=5)

        # Load images
        self.load_avatar_images()

    def create_settings_tab(self):
        """Create the settings tab"""
        # Create scrollable frame
        canvas = tk.Canvas(self.settings_frame)
        scrollbar = ttk.Scrollbar(self.settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # API Keys Section
        api_frame = ttk.LabelFrame(scrollable_frame, text="API Keys")
        api_frame.pack(fill='x', padx=10, pady=5)

        # OpenAI
        ttk.Label(api_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.openai_entry = ttk.Entry(api_frame, width=50, show='*')
        self.openai_entry.grid(row=0, column=1, padx=5, pady=2)
        self.openai_entry.insert(0, self.config.get("openai_api_key"))
        ttk.Button(api_frame, text="Test", command=self.test_openai).grid(row=0, column=2, padx=5, pady=2)

        # ElevenLabs
        ttk.Label(api_frame, text="ElevenLabs API Key:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.elevenlabs_entry = ttk.Entry(api_frame, width=50, show='*')
        self.elevenlabs_entry.grid(row=1, column=1, padx=5, pady=2)
        self.elevenlabs_entry.insert(0, self.config.get("elevenlabs_api_key"))

        # Azure
        ttk.Label(api_frame, text="Azure TTS Key:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.azure_entry = ttk.Entry(api_frame, width=50, show='*')
        self.azure_entry.grid(row=2, column=1, padx=5, pady=2)
        self.azure_entry.insert(0, self.config.get("azure_tts_key"))

        ttk.Label(api_frame, text="Azure Region:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.azure_region_entry = ttk.Entry(api_frame, width=50)
        self.azure_region_entry.grid(row=3, column=1, padx=5, pady=2)
        self.azure_region_entry.insert(0, self.config.get("azure_tts_region"))

        # OpenAI Model selection
        ttk.Label(api_frame, text="OpenAI Model:").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        # Define a few common model options; these can be expanded as desired
        model_options = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        self.openai_model_var = tk.StringVar(value=self.config.get("openai_model", "gpt-4o-mini"))
        model_combo = ttk.Combobox(api_frame, textvariable=self.openai_model_var, values=model_options)
        model_combo.grid(row=4, column=1, padx=5, pady=2, sticky='w')

        # Voice Settings
        voice_frame = ttk.LabelFrame(scrollable_frame, text="Voice Settings")
        voice_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(voice_frame, text="Voice Type:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.voice_var = tk.StringVar(value=self.config.get("ai_voice"))
        voice_combo = ttk.Combobox(voice_frame, textvariable=self.voice_var,
                                   values=["streamelements_brian", "elevenlabs", "azure", "pyttsx3"])
        voice_combo.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(voice_frame, text="ElevenLabs Voice Name:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.elevenlabs_voice_entry = ttk.Entry(voice_frame, width=30)
        self.elevenlabs_voice_entry.grid(row=1, column=1, padx=5, pady=2)
        self.elevenlabs_voice_entry.insert(0, self.config.get("elevenlabs_voice_name"))

        # Avatar Images
        avatar_frame = ttk.LabelFrame(scrollable_frame, text="Avatar Images")
        avatar_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(avatar_frame, text="Idle Image:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.idle_image_label = ttk.Label(avatar_frame, text=self.config.get("idle_image"))
        self.idle_image_label.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(avatar_frame, text="Browse", command=lambda: self.browse_image("idle")).grid(row=0, column=2, padx=5,
                                                                                                pady=2)

        ttk.Label(avatar_frame, text="Speaking Image:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.speaking_image_label = ttk.Label(avatar_frame, text=self.config.get("speaking_image"))
        self.speaking_image_label.grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(avatar_frame, text="Browse", command=lambda: self.browse_image("speaking")).grid(row=1, column=2,
                                                                                                    padx=5, pady=2)

        # Hotkeys
        hotkey_frame = ttk.LabelFrame(scrollable_frame, text="Hotkeys")
        hotkey_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(hotkey_frame, text="Start Voice Key:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.voice_key_entry = ttk.Entry(hotkey_frame, width=10)
        self.voice_key_entry.grid(row=0, column=1, padx=5, pady=2)
        self.voice_key_entry.insert(0, self.config.get("voice_key", "v"))

        ttk.Label(hotkey_frame, text="Stop Voice Key:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.stop_voice_key_entry = ttk.Entry(hotkey_frame, width=10)
        self.stop_voice_key_entry.grid(row=1, column=1, padx=5, pady=2)
        self.stop_voice_key_entry.insert(0, self.config.get("stop_voice_key", "p"))

        # Auto Commentary
        auto_frame = ttk.LabelFrame(scrollable_frame, text="Auto Commentary")
        auto_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(auto_frame, text="Commentary Interval (seconds):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.interval_spinbox = ttk.Spinbox(auto_frame, from_=5, to=300, width=10)
        self.interval_spinbox.grid(row=0, column=1, padx=5, pady=2)
        self.interval_spinbox.set(self.config.get("commentary_interval", 15))

        # System Prompt
        prompt_frame = ttk.LabelFrame(scrollable_frame, text="AI System Prompt")
        prompt_frame.pack(fill='x', padx=10, pady=5)

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=5, width=60)
        self.prompt_text.pack(padx=5, pady=5)
        self.prompt_text.insert('1.0', self.config.get("system_prompt"))

        # Save button
        save_btn = ttk.Button(scrollable_frame, text="Save All Settings", command=self.save_settings)
        save_btn.pack(pady=10)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_log_tab(self):
        """Create the chat/log tab"""
        # Chat display
        self.chat_text = scrolledtext.ScrolledText(self.log_frame, height=20, width=80)
        self.chat_text.pack(fill='both', expand=True, padx=10, pady=5)

        # Input frame
        input_frame = ttk.Frame(self.log_frame)
        input_frame.pack(fill='x', padx=10, pady=5)

        self.chat_entry = ttk.Entry(input_frame)
        self.chat_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))

        ttk.Button(input_frame, text="Send", command=self.send_chat_message).pack(side='right')

        # Clear log button
        ttk.Button(input_frame, text="Clear Log", command=self.clear_log).pack(side='right', padx=5)

    def show_setup_wizard(self):
        """Show initial setup wizard"""
        wizard = tk.Toplevel(self.root)
        wizard.title("Setup Wizard")
        wizard.geometry("600x500")
        wizard.transient(self.root)

        # Title
        title_label = tk.Label(wizard, text="Welcome to AI Streamer Assistant!", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        # Instructions
        instructions = tk.Label(wizard,
                                text="Let's set up your AI assistant. You'll need at least an OpenAI API key to get started.",
                                wraplength=500)
        instructions.pack(pady=10)

        # Setup frame
        setup_frame = ttk.Frame(wizard)
        setup_frame.pack(padx=20, pady=10, fill='both', expand=True)

        # OpenAI Setup
        openai_frame = ttk.LabelFrame(setup_frame, text="Step 1: OpenAI (Required)")
        openai_frame.pack(fill='x', pady=10)

        tk.Label(openai_frame,
                 text="1. Go to https://platform.openai.com/api-keys\n2. Create an account or sign in\n3. Generate a new API key\n4. Paste it below:",
                 justify='left').pack(padx=10, pady=5)

        openai_key_entry = ttk.Entry(openai_frame, width=50)
        openai_key_entry.pack(padx=10, pady=5)

        # OBS Setup info
        obs_frame = ttk.LabelFrame(setup_frame, text="Step 2: OBS Setup")
        obs_frame.pack(fill='x', pady=10)

        tk.Label(obs_frame,
                 text=f"1. Open OBS Studio\n2. Add an 'Image Source'\n3. Browse to: {os.path.abspath(OBS_OUTPUT_FILE)}\n4. Check 'Unload image when not showing' for smooth transitions",
                 justify='left').pack(padx=10, pady=5)

        # Images info
        images_frame = ttk.LabelFrame(setup_frame, text="Step 3: Avatar Images")
        images_frame.pack(fill='x', pady=10)

        tk.Label(images_frame,
                 text="Prepare two PNG images with transparency:\n• idle.png - Shows when AI is not speaking\n• speaking.png - Shows when AI is speaking\n\nYou can set these up after clicking Finish.",
                 justify='left').pack(padx=10, pady=5)

        # Buttons
        button_frame = ttk.Frame(wizard)
        button_frame.pack(pady=10)

        def finish_setup():
            api_key = openai_key_entry.get().strip()
            if api_key:
                # Save the API key
                if self.config.set("openai_api_key", api_key):
                    # Update the entry field in settings
                    self.openai_entry.delete(0, tk.END)
                    self.openai_entry.insert(0, api_key)
                    # Reinitialize AI with new key
                    self.ai.init_ai()
                    messagebox.showinfo("Success", "Setup complete! You can now use the AI assistant.")
                    wizard.destroy()
                else:
                    messagebox.showerror("Error", "Failed to save API key. Please try again.")
            else:
                messagebox.showwarning("Missing Key", "Please enter an OpenAI API key.")

        ttk.Button(button_frame, text="Finish Setup", command=finish_setup).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Skip", command=wizard.destroy).pack(side='left', padx=5)

    def toggle_obs_mode(self):
        """Toggle OBS output mode"""
        self.config.set("obs_mode", self.obs_mode_var.get())
        if self.obs_mode_var.get():
            self.obs_output.update_obs_image("idle")
            self.status_label.config(text="OBS output enabled")
        else:
            self.status_label.config(text="OBS output disabled")

    def test_openai(self):
        """Test OpenAI API connection"""
        api_key = self.openai_entry.get().strip()
        if api_key:
            self.config.set("openai_api_key", api_key)
            self.ai.init_ai()
            response = self.ai.get_response("Say 'Hello, I'm working!' if you can hear me.")
            if "working" in response.lower():
                messagebox.showinfo("Success", "OpenAI API is working!")
            else:
                messagebox.showwarning("Test Failed", f"Response: {response}")
        else:
            messagebox.showwarning("No API Key", "Please enter an OpenAI API key first.")

    def clear_log(self):
        """Clear the chat log"""
        self.chat_text.delete('1.0', tk.END)

    def browse_image(self, image_type):
        """Browse for an image file"""
        filename = filedialog.askopenfilename(
            title=f"Select {image_type} image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if filename:
            if image_type == "idle":
                self.config.set("idle_image", filename)
                self.idle_image_label.config(text=os.path.basename(filename))
            else:
                self.config.set("speaking_image", filename)
                self.speaking_image_label.config(text=os.path.basename(filename))
            self.load_avatar_images()
            # Update OBS output
            self.obs_output.update_obs_image(self.current_state if hasattr(self, 'current_state') else "idle")

    def load_avatar_images(self):
        """Load and prepare avatar images"""
        try:
            # Load idle image
            idle_path = self.config.get("idle_image")
            if idle_path and os.path.exists(idle_path):
                idle_img = Image.open(idle_path)
                idle_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                self.idle_photo = ImageTk.PhotoImage(idle_img)
            else:
                self.idle_photo = None

            # Load speaking image
            speaking_path = self.config.get("speaking_image")
            if speaking_path and os.path.exists(speaking_path):
                speaking_img = Image.open(speaking_path)
                speaking_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                self.speaking_photo = ImageTk.PhotoImage(speaking_img)
            else:
                self.speaking_photo = None

            # Show idle image by default
            self.show_idle_image()
        except Exception as e:
            print(f"Error loading images: {e}")

    def show_idle_image(self):
        """Display the idle image"""
        if self.idle_photo:
            self.avatar_label.config(image=self.idle_photo, text="")
        else:
            self.avatar_label.config(image="", text="No idle image")
        self.is_speaking = False
        self.current_state = "idle"
        self.obs_output.update_obs_image("idle")

    def show_speaking_image(self):
        """Display the speaking image"""
        if self.speaking_photo:
            self.avatar_label.config(image=self.speaking_photo, text="")
        else:
            self.avatar_label.config(image="", text="No speaking image")
        self.is_speaking = True
        self.current_state = "speaking"
        self.obs_output.update_obs_image("speaking")

    def save_settings(self):
        """Save all settings"""
        # Save all entries
        self.config.set("openai_api_key", self.openai_entry.get())
        self.config.set("elevenlabs_api_key", self.elevenlabs_entry.get())
        self.config.set("azure_tts_key", self.azure_entry.get())
        self.config.set("azure_tts_region", self.azure_region_entry.get())
        self.config.set("ai_voice", self.voice_var.get())
        self.config.set("elevenlabs_voice_name", self.elevenlabs_voice_entry.get())
        self.config.set("voice_key", self.voice_key_entry.get())
        self.config.set("stop_voice_key", self.stop_voice_key_entry.get())
        self.config.set("commentary_interval", int(self.interval_spinbox.get()))
        self.config.set("system_prompt", self.prompt_text.get('1.0', 'end-1c'))

        # Save OpenAI model selection
        if hasattr(self, 'openai_model_var'):
            self.config.set("openai_model", self.openai_model_var.get())

        # Reinitialize services with new settings
        self.ai.init_ai()
        self.stt.init_stt()

        messagebox.showinfo("Settings", "All settings saved successfully!")

    def toggle_voice_input(self):
        """Toggle voice input on/off"""
        if not self.is_listening:
            self.is_listening = True
            self.voice_btn.config(text="Stop Voice (P)")
            self.status_label.config(text="Listening... Press P to stop")

            # Start listening in a thread
            def listen_thread():
                text = self.stt.listen(self.config.get("stop_voice_key", "p"))
                if text and text.strip():
                    # Capture screen
                    screen_b64 = ScreenCapture.capture_screen_base64()
                    # Get AI response
                    prompt = f"User said: {text}"
                    if screen_b64:
                        prompt += " (Also consider what's visible on screen)"
                    response = self.ai.get_response(prompt, screen_b64)
                    self.message_queue.put(("user", text))
                    self.message_queue.put(("ai", response))

                self.is_listening = False
                self.root.after(0, lambda: self.voice_btn.config(text="Start Voice Input (V)"))
                self.root.after(0, lambda: self.status_label.config(text="Ready"))

            thread = threading.Thread(target=listen_thread)
            thread.daemon = True
            thread.start()
        else:
            keyboard.press_and_release(self.config.get("stop_voice_key", "p"))

    def toggle_auto_commentary(self):
        """Toggle automatic commentary"""
        self.auto_commentary_active = not self.auto_commentary_active
        if self.auto_commentary_active:
            self.auto_btn.config(text="Stop Auto Commentary")
            self.status_label.config(text="Auto commentary active")
            self.start_auto_commentary()
        else:
            self.auto_btn.config(text="Start Auto Commentary")
            self.status_label.config(text="Auto commentary stopped")

    def start_auto_commentary(self):
        """Start automatic commentary thread.

        The thread periodically captures the screen and obtains a short AI
        commentary. To allow the user to stop auto commentary promptly,
        the sleep interval is broken into small increments, checking
        whether the feature has been disabled after each increment.
        """

        def commentary_thread():
            # Read the interval at the start of the loop to avoid grabbing it repeatedly
            interval = self.config.get("commentary_interval", 15)
            while self.auto_commentary_active:
                # Sleep in 0.5‑second increments so that deactivation takes effect quickly
                waited = 0.0
                while waited < interval and self.auto_commentary_active:
                    time.sleep(0.5)
                    waited += 0.5
                # If auto commentary was deactivated during the wait, exit
                if not self.auto_commentary_active:
                    break
                # Only comment if not currently speaking or listening
                if not self.is_speaking and not self.is_listening:
                    # Capture screen
                    screen_b64 = ScreenCapture.capture_screen_base64()
                    if screen_b64:
                        # Get AI commentary
                        response = self.ai.get_response(
                            "Comment briefly on what you see happening on screen (1-2 sentences max)",
                            screen_b64
                        )
                        # Queue the response for processing in the GUI thread
                        self.message_queue.put(("ai", response))

        thread = threading.Thread(target=commentary_thread)
        thread.daemon = True
        thread.start()

    def test_speech(self):
        """Test speech synthesis"""
        test_text = "Hello! This is a test of the text to speech system. Ideally the voice you've chosen should be playing here, and the image should change while you hear it."
        self.message_queue.put(("ai", test_text))

    def test_screen_capture(self):
        """Test screen capture"""
        screen_b64 = ScreenCapture.capture_screen_base64()
        if screen_b64:
            response = self.ai.get_response("What do you see on the screen?", screen_b64)
            self.message_queue.put(("ai", response))
        else:
            messagebox.showwarning("Screen Capture", "Failed to capture screen")

    def send_chat_message(self):
        """Send a chat message to the AI"""
        message = self.chat_entry.get().strip()
        if message:
            self.chat_entry.delete(0, 'end')
            self.message_queue.put(("user", message))

            # Get AI response
            response = self.ai.get_response(message)
            self.message_queue.put(("ai", response))

    def process_messages(self):
        """Process messages from the queue"""
        try:
            while True:
                sender, message = self.message_queue.get_nowait()

                # Add to chat log
                timestamp = time.strftime("%H:%M:%S")
                self.chat_text.insert('end', f"[{timestamp}] {sender.upper()}: {message}\n")
                self.chat_text.see('end')

                # If AI message, speak it
                if sender == "ai":
                    self.show_speaking_image()
                    self.tts.speak(message, callback=self.show_idle_image)
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_messages)

    def run(self):
        """Start the application"""

        # Set up keyboard listener for voice input
        def keyboard_listener():
            while True:
                try:
                    voice_key = self.config.get("voice_key", "v")
                    if keyboard.is_pressed(voice_key) and not self.is_listening:
                        self.root.after(0, self.toggle_voice_input)
                        time.sleep(0.5)  # Debounce
                except:
                    pass
                time.sleep(0.1)

        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()

        self.root.mainloop()


if __name__ == "__main__":
    app = AIStreamerGUI()
    app.run()
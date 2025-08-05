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

DEFAULT_CONFIG = {
    "openai_api_key": "",
    "elevenlabs_api_key": "",
    "azure_tts_key": "",
    "azure_tts_region": "eastus",
    "twitch_username": "",
    "twitch_oauth": "",
    "twitch_channel": "",
    "ai_voice": "streamelements_brian",
    "elevenlabs_voice_name": "",
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
    "twitch_cooldown": 5  # Seconds between Twitch responses
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
        self.init_tts()

    def init_tts(self):
        """Initialize TTS engine based on availability"""
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=48000, buffer=1024)

        if PYTTSX3_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            voices = self.engine.getProperty('voices')
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
            self._speak_pyttsx3(text, callback)

    def _speak_streamelements(self, text, callback):
        """Use StreamElements Brian voice"""

        def speak_thread():
            audio_file = StreamElementsTTS.generate_speech(text, "Brian")
            if audio_file and PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Error playing StreamElements audio: {e}")
                    self._speak_pyttsx3(text, None)
                finally:
                    try:
                        if audio_file:
                            os.remove(audio_file)
                    except:
                        pass
                if callback:
                    callback()
            else:
                self._speak_pyttsx3(text, callback)

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
                    temp_file = "temp_audio.mp3"
                    save(audio, temp_file)

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
                self._speak_streamelements(text, callback)
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
                self._speak_streamelements(text, callback)
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

            while len(self.chat_history) > 20:
                if self.chat_history[1]["role"] != "system":
                    self.chat_history.pop(1)

            model_name = self.config.get("openai_model", "gpt-4o-mini")
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=self.chat_history,
                max_tokens=150
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
    """Main GUI Application"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lace's Simple TTS App")
        self.root.geometry("1200x1300")
        self.root.configure(bg=COLORS['bg'])

        # Set cute fonts
        self.setup_fonts()

        # Set style
        self.setup_styles()

        # Initialize managers
        self.config = ConfigManager()
        self.tts = TTSManager(self.config)
        self.ai = AIManager(self.config)
        self.obs_output = OBSImageOutput(self.config)

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
                        font=('Comic Sans MS', 10))
        style.configure('TRadiobutton', background=COLORS['frame_bg'], foreground=COLORS['text'])
        style.configure('TSpinbox', fieldbackground=COLORS['entry_bg'])
        style.configure('TCombobox', fieldbackground=COLORS['entry_bg'])

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
        response_frame = ttk.LabelFrame(self.twitch_frame, text="AI Response Settings")
        response_frame.pack(fill='x', padx=20, pady=10)

        settings_grid = ttk.Frame(response_frame)
        settings_grid.pack(padx=10, pady=10)

        self.respond_twitch_var = tk.BooleanVar(value=self.config.get("respond_to_twitch", True))
        ttk.Checkbutton(settings_grid, text="Enable AI responses to chat",
                        variable=self.respond_twitch_var).grid(row=0, column=0, sticky='w', pady=5)

        ttk.Label(settings_grid, text="Response chance (%):", font=self.small_font).grid(row=1, column=0, sticky='w',
                                                                                         pady=5)
        self.response_chance_var = tk.IntVar(value=int(self.config.get("twitch_response_chance", 0.8) * 100))
        response_scale = ttk.Scale(settings_grid, from_=0, to=100, orient='horizontal',
                                   variable=self.response_chance_var, length=200)
        response_scale.grid(row=1, column=1, pady=5)
        self.response_chance_label = ttk.Label(settings_grid, text=f"{self.response_chance_var.get()}%")
        self.response_chance_label.grid(row=1, column=2, padx=10)

        # Update label when scale moves
        response_scale.configure(command=lambda v: self.response_chance_label.config(text=f"{int(float(v))}%"))

        ttk.Label(settings_grid, text="Cooldown (seconds):", font=self.small_font).grid(row=2, column=0, sticky='w',
                                                                                        pady=5)
        self.cooldown_spinbox = ttk.Spinbox(settings_grid, from_=1, to=60, width=10)
        self.cooldown_spinbox.grid(row=2, column=1, sticky='w', pady=5)
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

        # API Keys Card
        api_card = ttk.LabelFrame(settings_container, text="🔑 API Keys")
        api_card.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

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

        # Voice Settings Card
        voice_card = ttk.LabelFrame(settings_container, text="🎤 Voice Settings")
        voice_card.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        ttk.Label(voice_card, text="Voice Type:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.voice_var = tk.StringVar(value=self.config.get("ai_voice"))
        voice_combo = ttk.Combobox(voice_card, textvariable=self.voice_var, width=20,
                                   values=["streamelements_brian", "elevenlabs", "azure", "pyttsx3"])
        voice_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(voice_card, text="Push-to-Talk Key:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.ptt_key_entry = ttk.Entry(voice_card, width=10)
        self.ptt_key_entry.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        self.ptt_key_entry.insert(0, self.config.get("push_to_talk_key", "v"))

        # Twitch Settings Card
        twitch_card = ttk.LabelFrame(settings_container, text="💬 Twitch Settings")
        twitch_card.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

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
        avatar_card.grid(row=1, column=1, sticky='ew', padx=5, pady=5)

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

        # System Prompt Card
        prompt_card = ttk.LabelFrame(settings_container, text="💭 AI Personality")
        prompt_card.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        self.prompt_text = scrolledtext.ScrolledText(prompt_card, height=4, width=60,
                                                     bg=COLORS['entry_bg'], fg=COLORS['text'],
                                                     font=self.small_font)
        self.prompt_text.pack(padx=10, pady=10)
        self.prompt_text.insert('1.0', self.config.get("system_prompt"))

        # Save Button
        save_frame = ttk.Frame(settings_container)
        save_frame.grid(row=3, column=0, columnspan=2, pady=20)

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

                # Respond if enabled
                if self.respond_twitch_var.get() and self.twitch_connected:
                    self.process_twitch_message_for_ai(username, message)
            else:
                self.twitch_chat_display.insert('end', f"{content}\n", 'message')

        self.twitch_chat_display.see('end')

    def process_twitch_message_for_ai(self, username, message):
        """Process Twitch message and generate AI response"""
        import random

        # Check cooldown
        current_time = time.time()
        cooldown = self.config.get("twitch_cooldown", 5)

        if current_time - self.twitch.last_response_time < cooldown:
            return

        # Check response chance
        chance = self.response_chance_var.get() / 100.0
        if random.random() > chance:
            return

        # Generate AI response
        response = self.ai.get_response(
            f"Twitch viewer {username} said: {message}. Respond briefly and conversationally."
        )

        # Update last response time
        self.twitch.last_response_time = current_time

        # Add response to message queue for TTS
        self.message_queue.put(("ai", response))

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
        """Show beautiful setup wizard"""
        wizard = tk.Toplevel(self.root)
        wizard.title("Welcome to Lace's Simple TTS App!")
        wizard.geometry("700x600")
        wizard.configure(bg=COLORS['bg'])
        wizard.transient(self.root)

        # Header
        header = tk.Frame(wizard, bg=COLORS['accent'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(header, text="✨ Welcome! ✨",
                 font=self.title_font, bg=COLORS['accent'], fg='white').pack(expand=True)

        # Content
        content = ttk.Frame(wizard)
        content.pack(fill='both', expand=True, padx=30, pady=20)

        # Step 1
        step1_frame = ttk.LabelFrame(content, text="Step 1: OpenAI Setup (Required)")
        step1_frame.pack(fill='x', pady=10)

        ttk.Label(step1_frame, text="Get your API key from: https://platform.openai.com/api-keys",
                  foreground=COLORS['info']).pack(padx=10, pady=5)

        self.wizard_api_entry = ttk.Entry(step1_frame, width=50, font=self.normal_font)
        self.wizard_api_entry.pack(padx=10, pady=10)

        # Step 2
        step2_frame = ttk.LabelFrame(content, text="Step 2: Choose Your Voice")
        step2_frame.pack(fill='x', pady=10)

        ttk.Label(step2_frame, text="StreamElements Brian is free and works immediately!").pack(padx=10, pady=10)

        # Step 3
        step3_frame = ttk.LabelFrame(content, text="Step 3: Twitch (Optional)")
        step3_frame.pack(fill='x', pady=10)

        ttk.Label(step3_frame, text="You can connect to Twitch later in the Settings tab!").pack(padx=10, pady=10)

        # Buttons
        button_frame = ttk.Frame(wizard)
        button_frame.pack(pady=20)

        def finish_setup():
            api_key = self.wizard_api_entry.get().strip()
            if api_key:
                self.config.set("openai_api_key", api_key)
                self.ai.init_ai()
                messagebox.showinfo("Success", "Setup complete! Press V to start talking!")
                wizard.destroy()
            else:
                messagebox.showwarning("Missing Key", "Please enter an OpenAI API key.")

        StyledButton(button_frame, text="Complete Setup", command=finish_setup,
                     bg=COLORS['success']).pack(side='left', padx=10)
        StyledButton(button_frame, text="Skip for Now", command=wizard.destroy).pack(side='left')

    def setup_push_to_talk(self):
        """Setup push-to-talk functionality"""
        # This would need actual speech recognition implementation
        # For now, using placeholder
        pass

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

                if not self.is_speaking and not self.is_listening:
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
        self.config.set("openai_api_key", self.openai_entry.get())
        self.config.set("openai_model", self.model_var.get())
        self.config.set("ai_voice", self.voice_var.get())
        self.config.set("push_to_talk_key", self.ptt_key_entry.get())
        self.config.set("twitch_username", self.twitch_username_entry.get())
        self.config.set("twitch_oauth", self.twitch_oauth_entry.get())
        self.config.set("twitch_channel", self.twitch_channel_entry.get())
        self.config.set("respond_to_twitch", self.respond_twitch_var.get())
        self.config.set("twitch_response_chance", self.response_chance_var.get() / 100.0)
        self.config.set("twitch_cooldown", int(self.cooldown_spinbox.get()))
        self.config.set("commentary_interval",
                        int(self.interval_spinbox.get()) if hasattr(self, 'interval_spinbox') else 15)
        self.config.set("system_prompt", self.prompt_text.get('1.0', 'end-1c'))

        self.ptt_label.config(text=f"Hold [{self.ptt_key_entry.get().upper()}] to talk")

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

                elif sender == "user":
                    self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
                    self.chat_text.insert('end', f"YOU: ", 'user')
                    self.chat_text.insert('end', f"{message}\n", 'user_message')

                elif sender == "twitch":
                    self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
                    self.chat_text.insert('end', f"TWITCH: ", 'twitch')
                    self.chat_text.insert('end', f"{message}\n", 'twitch_message')

                # Configure text tags
                self.chat_text.tag_config('timestamp', foreground='#7F8C8D')
                self.chat_text.tag_config('ai', foreground='#E74C3C', font=('Arial', 10, 'bold'))
                self.chat_text.tag_config('ai_message', foreground='#ECF0F1')
                self.chat_text.tag_config('user', foreground='#3498DB', font=('Arial', 10, 'bold'))
                self.chat_text.tag_config('user_message', foreground='#ECF0F1')
                self.chat_text.tag_config('twitch', foreground='#9B59B6', font=('Arial', 10, 'bold'))
                self.chat_text.tag_config('twitch_message', foreground='#ECF0F1')

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
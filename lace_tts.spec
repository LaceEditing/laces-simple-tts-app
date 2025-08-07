# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification file for Lace's Simple TTS App.

This spec aims to create a one‑file Windows executable that includes all
required data files (images and icon) and bundles native dependencies such as
the Azure Speech SDK.  It collects all binaries, data files, and hidden
imports required by `azure.cognitiveservices.speech` using PyInstaller's
`collect_all` helper.  Additional hidden imports for optional packages like
`pygame`, `gtts`, `pyttsx3`, and `pyaudio` are listed explicitly to ensure
they are detected at build time.
"""

import os
from PyInstaller.utils.hooks import collect_all

# Collect all binaries, data files, and hidden imports for the Azure Speech SDK.
azure_datas, azure_binaries, azure_hiddenimports = collect_all('azure.cognitiveservices.speech')

# Analysis: describe the main script and include data/binary resources.
a = Analysis(
    ['main.py'],
    # Use the current working directory as the search path.  __file__ is not
    # defined when the spec is executed by PyInstaller, so avoid using it here.
    pathex=[os.getcwd()],
    binaries=azure_binaries,
    datas=[
        ('current_avatar.png', '.'),
        ('idle.png', '.'),
        ('speaking.png', '.'),
        ('app_icon.ico', '.'),
    ] + azure_datas,
    hiddenimports=[
        'pygame',
        'gtts',
        'pyttsx3',
        'pyaudio',
    ] + azure_hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Create a PYZ archive of the pure python modules.
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Build the executable in one‑file mode.  The absence of a COLLECT call and
# setting `onefile=True` instructs PyInstaller to bundle everything into a
# single executable.  `console=False` hides the console window.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LaceTTS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='app_icon.ico',
    onefile=True,
)
#!/usr/bin/env python3
"""
Audio manager for Magic Mirror Face Filter.
Handles background music playback for different filters.
"""

import os
import subprocess
import threading
from typing import Optional, Dict

# Try to import pygame for audio, fallback to command-line player
try:
    import pygame

    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Note: pygame not installed. Using system audio player.")


class AudioManager:
    """Manages audio playback for filters."""

    def __init__(self, sounds_dir: str = "assets/sounds"):
        """
        Initialize audio manager.

        Args:
            sounds_dir: Directory containing sound files
        """
        self.sounds_dir = sounds_dir
        self.current_sound: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None

        # Map filter names to sound files
        self.filter_sounds: Dict[str, str] = {
            "Clown Nose": "christmas.mp3",
            "Unicorn Horn": "unicorn.mp3",
        }

        # Ensure sounds directory exists
        os.makedirs(sounds_dir, exist_ok=True)

    def play_for_filter(self, filter_name: str):
        """
        Play appropriate sound for the given filter.

        Args:
            filter_name: Name of the current filter
        """
        sound_file = self.filter_sounds.get(filter_name)

        if sound_file:
            sound_path = os.path.join(self.sounds_dir, sound_file)
            if os.path.exists(sound_path):
                self._play_sound(sound_path)
            else:
                print(f"Sound file not found: {sound_path}")
                self.stop()
        else:
            # No sound for this filter, stop any playing
            self.stop()

    def _play_sound(self, sound_path: str):
        """Play a sound file."""
        # Don't restart if already playing same sound
        if self.current_sound == sound_path:
            return

        self.stop()
        self.current_sound = sound_path

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play(-1)  # -1 = loop forever
            except Exception as e:
                print(f"Error playing sound with pygame: {e}")
        else:
            # Fallback to command-line player
            self._play_with_system(sound_path)

    def _play_with_system(self, sound_path: str):
        """Play sound using system command (Linux/Raspberry Pi)."""
        try:
            # Try different players available on Raspberry Pi
            for player in ["mpg123", "ffplay", "aplay", "paplay"]:
                try:
                    if player == "ffplay":
                        cmd = [player, "-nodisp", "-autoexit", "-loop", "0", sound_path]
                    elif player == "mpg123":
                        cmd = [player, "--loop", "-1", "-q", sound_path]
                    else:
                        cmd = [player, sound_path]

                    self._process = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    return
                except FileNotFoundError:
                    continue

            print("No audio player found. Install mpg123: sudo apt install mpg123")
        except Exception as e:
            print(f"Error playing sound: {e}")

    def stop(self):
        """Stop any playing sound."""
        self.current_sound = None

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except:
                pass

        if self._process:
            try:
                self._process.terminate()
                self._process = None
            except:
                pass

    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except:
                pass

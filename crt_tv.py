#!/usr/bin/env python3
"""
CRT Television Simulator
A retro-styled video player with authentic CRT aesthetics and physical controls.
"""

import pygame
import cv2
import os
import random
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time


# ==================== CONFIGURATION ====================

@dataclass(frozen=True)
class Config:
    """Application configuration constants."""
    # Display
    SCREEN_W: int = 960
    SCREEN_H: int = 720
    FPS: int = 60

    # CRT Layout
    CRT_MARGIN: int = 80
    PANEL_HEIGHT: int = 70

    # Video
    VIDEO_DIR: str = "videos"
    DEFAULT_FPS: float = 30.0

    # Audio
    FFPLAY_PATH: str = "ffplay"
    DEFAULT_VOLUME: float = 0.7
    VOLUME_STEP: float = 0.1

    # Effects
    BLACKOUT_DURATION: float = 0.2

    # Colors
    COLOR_TV_BODY: Tuple[int, int, int] = (30, 30, 30)
    COLOR_PANEL: Tuple[int, int, int] = (22, 22, 22)
    COLOR_SCREEN_OFF: Tuple[int, int, int] = (45, 50, 45)
    COLOR_LED_ON: Tuple[int, int, int] = (60, 210, 60)
    COLOR_LED_OFF: Tuple[int, int, int] = (190, 40, 40)
    COLOR_TEXT: Tuple[int, int, int] = (160, 160, 160)
    COLOR_TEXT_BRIGHT: Tuple[int, int, int] = (230, 230, 230)
    COLOR_BUTTON_FACE: Tuple[int, int, int] = (60, 60, 60)
    COLOR_BUTTON_EDGE: Tuple[int, int, int] = (10, 10, 10)
    COLOR_VOL_FACE: Tuple[int, int, int] = (70, 70, 70)
    COLOR_VOL_EDGE: Tuple[int, int, int] = (15, 15, 15)


class TVState(Enum):
    """Television operating states."""
    OFF = "off"
    ON = "on"
    BLACKOUT = "blackout"


# ==================== VIDEO MANAGER ====================

class VideoManager:
    """Handles video file loading and frame extraction."""

    def __init__(self, video_dir: str, default_fps: float):
        self.video_dir = Path(video_dir)
        self.default_fps = default_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_path: Optional[Path] = None
        self.fps: float = default_fps
        self.frame_duration: float = 1.0 / default_fps
        self.time_accumulator: float = 0.0
        self.last_frame: Optional[np.ndarray] = None

    def get_available_videos(self) -> List[Path]:
        """Get list of available video files."""
        if not self.video_dir.exists():
            return []
        exts = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
        return [p for p in self.video_dir.iterdir()
                if p.is_file() and p.suffix.lower() in exts]

    def load_video(self, video_path: Path) -> bool:
        """Load a video file for playback."""
        try:
            # Release previous video
            if self.cap:
                self.cap.release()

            # Open new video
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return False

            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps > 0 else self.default_fps
            self.frame_duration = 1.0 / self.fps

            # Reset state
            self.current_path = video_path
            self.time_accumulator = 0.0
            self.last_frame = None

            return True

        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def load_random_video(self) -> bool:
        """Load a random video from the video directory."""
        videos = self.get_available_videos()
        if not videos:
            print(f"No videos found in {self.video_dir}")
            return False

        video_path = random.choice(videos)
        return self.load_video(video_path)

    def update(self, dt: float) -> Optional[np.ndarray]:
        """Update video playback and return current frame."""
        if not self.cap:
            return None

        self.time_accumulator += dt

        # Process frames based on time accumulation
        while self.time_accumulator >= self.frame_duration:
            ret, frame = self.cap.read()

            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.time_accumulator = 0.0
                return self.last_frame

            self.last_frame = frame
            self.time_accumulator -= self.frame_duration

        return self.last_frame

    def release(self):
        """Release video resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_path = None
        self.last_frame = None


# ==================== AUDIO MANAGER ====================

class AudioManager:
    """Handles audio playback via ffplay subprocess with volume control."""

    def __init__(self, ffplay_path: str):
        self.ffplay_path = ffplay_path
        self.process: Optional[subprocess.Popen] = None
        self.current_video: Optional[Path] = None
        self._volume: float = 0.7
        self._pending_volume_change: bool = False

    @property
    def volume(self) -> float:
        """Get current volume (0.0 - 1.0)."""
        return self._volume

    @volume.setter
    def volume(self, value: float):
        """Set volume - marks for change but doesn't restart playback."""
        new_volume = max(0.0, min(1.0, value))
        if new_volume != self._volume:
            self._volume = new_volume
            # Note: Volume will take effect on next channel change
            # This prevents audio restart mid-playback

    def play(self, video_path: Path) -> bool:
        """Start audio playback for a video file."""
        try:
            # Stop current audio
            self.stop()

            # Start new audio process with current volume
            self.process = subprocess.Popen(
                [
                    self.ffplay_path,
                    "-nodisp",  # No video display
                    "-vn",  # No video decoding
                    "-loglevel", "quiet",  # Suppress logs
                    "-fflags", "nobuffer",  # No buffering
                    "-flags", "low_delay",  # Low latency
                    "-loop", "-1",  # Infinite loop
                    "-sync", "audio",  # Sync to audio
                    "-af", f"volume={self._volume}",  # Volume control
                    str(video_path)
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            self.current_video = video_path
            self._pending_volume_change = False
            return True

        except Exception as e:
            print(f"Error starting audio: {e}")
            return False

    def stop(self):
        """Stop audio playback."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                self.process = None

    def is_alive(self) -> bool:
        """Check if audio process is running."""
        return self.process is not None and self.process.poll() is None

    def restart_if_needed(self):
        """Restart audio if process died unexpectedly."""
        if self.current_video and not self.is_alive():
            self.play(self.current_video)


# ==================== UI ELEMENTS ====================

class Button:
    """Interactive button UI element."""

    def __init__(self, rect: pygame.Rect, face_color: Tuple[int, int, int],
                 edge_color: Tuple[int, int, int]):
        self.rect = rect
        self.face_color = face_color
        self.edge_color = edge_color

    def draw(self, surface: pygame.Surface):
        """Draw button with 3D effect."""
        pygame.draw.rect(surface, self.edge_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, self.face_color, self.rect.inflate(-4, -4),
                         border_radius=3)

    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within button bounds."""
        return self.rect.collidepoint(pos)


class ControlPanel:
    """TV control panel with buttons and indicators."""

    def __init__(self, config: Config):
        self.config = config
        panel_y = config.SCREEN_H - config.PANEL_HEIGHT

        # Create buttons
        self.power_button = Button(
            pygame.Rect(80, panel_y + 20, 90, 30),
            config.COLOR_BUTTON_FACE,
            config.COLOR_BUTTON_EDGE
        )

        self.vol_down_button = Button(
            pygame.Rect(220, panel_y + 20, 40, 30),
            config.COLOR_VOL_FACE,
            config.COLOR_VOL_EDGE
        )

        self.vol_up_button = Button(
            pygame.Rect(270, panel_y + 20, 40, 30),
            config.COLOR_VOL_FACE,
            config.COLOR_VOL_EDGE
        )

        self.panel_rect = pygame.Rect(0, panel_y, config.SCREEN_W,
                                      config.PANEL_HEIGHT)
        self.font = pygame.font.SysFont("arial", 14)

    def draw(self, surface: pygame.Surface, tv_on: bool, volume: float):
        """Draw control panel with volume indicator."""
        # Panel background
        pygame.draw.rect(surface, self.config.COLOR_PANEL, self.panel_rect)

        # Labels
        power_label = self.font.render("POWER", True, self.config.COLOR_TEXT)
        vol_label = self.font.render("VOL", True, self.config.COLOR_TEXT)

        surface.blit(power_label, (80, self.panel_rect.top + 2))
        surface.blit(vol_label, (235, self.panel_rect.top + 2))

        # Power button with LED
        self.power_button.draw(surface)
        led_color = self.config.COLOR_LED_ON if tv_on else self.config.COLOR_LED_OFF
        led_rect = self.power_button.rect.inflate(-18, -10)
        pygame.draw.rect(surface, led_color, led_rect, border_radius=3)

        # Volume buttons
        self.vol_down_button.draw(surface)
        self.vol_up_button.draw(surface)

        # Volume button labels
        minus = self.font.render("-", True, self.config.COLOR_TEXT_BRIGHT)
        plus = self.font.render("+", True, self.config.COLOR_TEXT_BRIGHT)

        minus_rect = minus.get_rect(center=self.vol_down_button.rect.center)
        plus_rect = plus.get_rect(center=self.vol_up_button.rect.center)

        surface.blit(minus, minus_rect)
        surface.blit(plus, plus_rect)

        # Volume bar visualization
        bar_x = 330
        bar_y = self.panel_rect.top + 25
        bar_width = 200
        bar_height = 20
        num_bars = 20
        bar_segment_width = (bar_width - (num_bars - 1) * 2) // num_bars

        filled_bars = int(volume * num_bars)

        for i in range(num_bars):
            x = bar_x + i * (bar_segment_width + 2)
            if i < filled_bars:
                # Filled bar - bright green
                color = (60, 255, 60)
            else:
                # Empty bar - dark gray
                color = (40, 40, 40)

            pygame.draw.rect(surface, color,
                             (x, bar_y, bar_segment_width, bar_height))


# ==================== CRT TELEVISION ====================

class CRTTelevision:
    """Main CRT television simulator."""

    def __init__(self, config: Config):
        self.config = config

        # Initialize Pygame
        pygame.init()
        pygame.mixer.init()

        # Create display
        self.screen = pygame.display.set_mode(
            (config.SCREEN_W, config.SCREEN_H),
            pygame.FULLSCREEN
        )
        pygame.display.set_caption("CRT Television")
        self.clock = pygame.time.Clock()

        # CRT screen area
        self.crt_rect = pygame.Rect(
            config.CRT_MARGIN,
            config.CRT_MARGIN,
            config.SCREEN_W - config.CRT_MARGIN * 2,
            config.SCREEN_H - config.CRT_MARGIN * 2 - config.PANEL_HEIGHT
        )

        # Load sounds
        self.load_sounds()

        # Initialize managers
        self.video_manager = VideoManager(config.VIDEO_DIR, config.DEFAULT_FPS)
        self.audio_manager = AudioManager(config.FFPLAY_PATH)
        self.audio_manager.volume = config.DEFAULT_VOLUME

        # Initialize UI
        self.control_panel = ControlPanel(config)

        # State
        self.state = TVState.OFF
        self.blackout_timer = 0.0
        self.running = True

    def load_sounds(self):
        """Load sound effects."""
        try:
            self.sound_on = pygame.mixer.Sound("tv_on.wav")
            self.sound_off = pygame.mixer.Sound("tv_off.wav")
            self.sound_on.set_volume(0.5)
            self.sound_off.set_volume(0.5)
        except Exception as e:
            print(f"Warning: Could not load sounds: {e}")
            # Create silent sounds as fallback
            self.sound_on = pygame.mixer.Sound(buffer=bytes(44100))
            self.sound_off = pygame.mixer.Sound(buffer=bytes(44100))

    def turn_on(self):
        """Turn TV on."""
        self.sound_on.play()
        self.state = TVState.ON

        if self.video_manager.load_random_video():
            if self.video_manager.current_path:
                self.audio_manager.play(self.video_manager.current_path)

    def turn_off(self):
        """Turn TV off."""
        self.sound_off.play()
        self.state = TVState.OFF
        self.video_manager.release()
        self.audio_manager.stop()

    def change_channel(self):
        """Switch to random channel with blackout effect."""
        if self.state != TVState.ON:
            return

        # Start blackout
        self.state = TVState.BLACKOUT
        self.blackout_timer = self.config.BLACKOUT_DURATION

        # Load new video
        if self.video_manager.load_random_video():
            if self.video_manager.current_path:
                self.audio_manager.play(self.video_manager.current_path)

    def handle_events(self):
        """Process user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r and self.state == TVState.ON:
                    self.change_channel()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Power button
                if self.control_panel.power_button.is_clicked(event.pos):
                    if self.state == TVState.OFF:
                        self.turn_on()
                    else:
                        self.turn_off()

                # Volume buttons
                elif self.control_panel.vol_down_button.is_clicked(event.pos):
                    self.audio_manager.volume -= self.config.VOLUME_STEP

                elif self.control_panel.vol_up_button.is_clicked(event.pos):
                    self.audio_manager.volume += self.config.VOLUME_STEP

    def update(self, dt: float):
        """Update TV state."""
        # Restart audio if it died
        if self.state == TVState.ON:
            self.audio_manager.restart_if_needed()

        # Update blackout timer
        if self.state == TVState.BLACKOUT:
            self.blackout_timer -= dt
            if self.blackout_timer <= 0:
                self.state = TVState.ON

    def render(self):
        """Render TV display."""
        # TV body background
        self.screen.fill(self.config.COLOR_TV_BODY)

        # CRT screen
        if self.state == TVState.ON:
            # Get current video frame
            frame = self.video_manager.update(self.clock.get_time() / 1000.0)

            if frame is not None:
                # Convert and resize frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, self.crt_rect.size)

                # Create pygame surface and display
                surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                self.screen.blit(surface, self.crt_rect.topleft)
            else:
                # No frame available - show black
                pygame.draw.rect(self.screen, (0, 0, 0), self.crt_rect)

        elif self.state == TVState.BLACKOUT:
            # Channel change blackout
            pygame.draw.rect(self.screen, (0, 0, 0), self.crt_rect)

        else:  # TVState.OFF
            # TV off - show dark green screen
            pygame.draw.rect(self.screen, self.config.COLOR_SCREEN_OFF,
                             self.crt_rect)

        # Draw control panel
        self.control_panel.draw(self.screen, self.state != TVState.OFF,
                                self.audio_manager.volume)

        # Update display
        pygame.display.flip()

    def run(self):
        """Main application loop."""
        while self.running:
            dt = self.clock.tick(self.config.FPS) / 1000.0

            self.handle_events()
            self.update(dt)
            self.render()

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.audio_manager.stop()
        self.video_manager.release()
        pygame.quit()


# ==================== MAIN ====================

def main():
    """Application entry point."""
    try:
        config = Config()
        tv = CRTTelevision(config)
        tv.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

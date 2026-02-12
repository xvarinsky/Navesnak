#!/usr/bin/env python3
"""
Magic Mirror Face Filter - Raspberry Pi Application

A fun, interactive app that shows a live mirrored video with face detection
and overlay filters (crown, butterfly wings, neon mask, fire eyes, halo).

Controls:
    Mouse/Touch - Click left/right sides to navigate filters
    ← →        - Previous/Next filter  
    1-4        - Select filter directly
    SPACE      - Next filter
    C          - Capture screenshot
    F          - Toggle fullscreen
    Q/ESC      - Quit
"""

import cv2
import argparse
import sys
import os
import time
import numpy as np
from datetime import datetime
from face_detector import FaceMeshReplacement
from filters import FilterManager, overlay_filter


class ModernUI:
    """Modern glassmorphism-style UI overlay."""
    
    def __init__(self):
        # Colors (BGR)
        self.accent_color = (255, 150, 50)  # Cyan accent
        self.secondary_color = (200, 100, 255)  # Magenta
        self.text_color = (255, 255, 255)
        self.shadow_color = (50, 50, 50)
        self.success_color = (100, 255, 100)
        
        # Animation state
        self.filter_change_time = 0
        self.face_detected = False
        self.pulse_phase = 0
        self.screenshot_flash = 0
        
    def draw_glassmorphism_panel(self, frame: np.ndarray, x: int, y: int, 
                                  width: int, height: int, alpha: float = 0.3):
        """Draw a frosted glass effect panel."""
        overlay = frame.copy()
        
        # Draw semi-transparent dark rectangle
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     (30, 30, 40), -1)
        
        # Add border glow
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     self.accent_color, 1)
        
        # Blend with original
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def draw_filter_carousel(self, frame: np.ndarray, filter_manager, 
                             current_index: int, frame_width: int, frame_height: int):
        """Draw the modern filter carousel at the bottom."""
        if filter_manager is None:
            return frame
            
        panel_height = 80
        panel_y = frame_height - panel_height - 20
        panel_margin = 40
        panel_width = frame_width - (panel_margin * 2)
        
        # Draw glass panel
        frame = self.draw_glassmorphism_panel(
            frame, panel_margin, panel_y, panel_width, panel_height, 0.5
        )
        
        # Get filter info
        filter_count = filter_manager.get_filter_count()
        filter_name = filter_manager.get_current_filter_name()
        
        # Draw filter name with animation
        time_since_change = time.time() - self.filter_change_time
        scale = 1.0 + max(0, 0.3 - time_since_change) * 0.5  # Pop animation
        
        # Main filter name
        text_size = cv2.getTextSize(filter_name, cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.9 * scale, 2)[0]
        text_x = panel_margin + (panel_width - text_size[0]) // 2
        text_y = panel_y + 35
        
        # Shadow
        cv2.putText(frame, filter_name, (text_x + 2, text_y + 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, self.shadow_color, 3)
        # Main text
        cv2.putText(frame, filter_name, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, self.text_color, 2)
        
        # Draw filter dots/indicators
        dot_y = panel_y + 55
        dot_spacing = 25
        total_dots_width = (filter_count - 1) * dot_spacing
        dot_start_x = panel_margin + (panel_width - total_dots_width) // 2
        
        for i in range(filter_count):
            dot_x = dot_start_x + i * dot_spacing
            if i == current_index:
                # Active dot - larger and colored
                cv2.circle(frame, (dot_x, dot_y), 8, self.accent_color, -1)
                cv2.circle(frame, (dot_x, dot_y), 8, self.text_color, 1)
            else:
                # Inactive dot
                cv2.circle(frame, (dot_x, dot_y), 5, (100, 100, 100), -1)
        
        # Draw navigation hints
        hint_y = panel_y + 70
        left_hint = "◀ B"
        right_hint = "SPACE ▶"
        
        cv2.putText(frame, left_hint, (panel_margin + 15, hint_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        right_text_size = cv2.getTextSize(right_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(frame, right_hint, (panel_margin + panel_width - right_text_size[0] - 15, hint_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        return frame
    
    def draw_face_indicator(self, frame: np.ndarray, face_detected: bool, frame_width: int):
        """Draw a pulsing face detection indicator."""
        self.face_detected = face_detected
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * np.pi)
        
        # Position in top-right
        indicator_x = frame_width - 60
        indicator_y = 30
        
        if face_detected:
            # Pulsing green circle
            pulse = int(5 * (1 + np.sin(self.pulse_phase)))
            color = self.success_color
            cv2.circle(frame, (indicator_x, indicator_y), 12 + pulse, color, 2)
            cv2.circle(frame, (indicator_x, indicator_y), 6, color, -1)
            
            # Text
            cv2.putText(frame, "FACE", (indicator_x - 25, indicator_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # Red static circle
            color = (80, 80, 200)
            cv2.circle(frame, (indicator_x, indicator_y), 12, color, 2)
            
            cv2.putText(frame, "NO FACE", (indicator_x - 35, indicator_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        return frame
    
    def draw_controls_hint(self, frame: np.ndarray, frame_width: int):
        """Draw minimal controls hint at top-left."""
        hints = "1-4: Select | C: Capture | F: Fullscreen | Q: Quit"
        
        cv2.putText(frame, hints, (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def draw_screenshot_flash(self, frame: np.ndarray):
        """Draw a flash effect when screenshot is taken."""
        if self.screenshot_flash > 0:
            alpha = min(0.5, self.screenshot_flash)
            cv2.addWeighted(frame, 1 - alpha, frame, 0, 255 * alpha, frame)
            self.screenshot_flash -= 0.1
        
        return frame
    
    def trigger_filter_change(self):
        """Trigger animation for filter change."""
        self.filter_change_time = time.time()
    
    def trigger_screenshot_flash(self):
        """Trigger screenshot flash effect."""
        self.screenshot_flash = 0.6


class MagicMirrorApp:
    """Main application class for Magic Mirror Face Filter."""

    def __init__(self, camera_index: int = 0, fullscreen: bool = True):
        """
        Initialize the application.

        Args:
            camera_index: Camera device index (0 for default, 1 for USB, etc.)
            fullscreen: Whether to start in fullscreen mode
        """
        self.camera_index = camera_index
        self.fullscreen = fullscreen
        self.cap = None
        self.filter_manager = None
        self.ui = ModernUI()
        self.screenshots_dir = os.path.expanduser("~/Desktop")

        # Initialize OpenCV DNN Face Detection (ARM64 compatible)
        self.face_mesh = FaceMeshReplacement(min_detection_confidence=0.5)

        # Frame skipping for performance - only run detection every N frames
        self._frame_count = 0
        self._detect_every_n = 3  # Run detection every 3rd frame
        self._last_face_landmarks = None
        self._last_face_detected = False

    def initialize_camera(self) -> bool:
        """Initialize the camera capture with multiple backend attempts."""
        backends_to_try = [
            # Try V4L2 backend first (works better on Raspberry Pi)
            (cv2.CAP_V4L2, "V4L2"),
            # Try default backend
            (cv2.CAP_ANY, "default"),
        ]

        # For Raspberry Pi Camera Module, try libcamera pipeline
        libcamera_pipelines = [
            "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink",
            f"v4l2src device=/dev/video{self.camera_index} ! video/x-raw,width=640,height=480 ! videoconvert ! appsink",
        ]

        print(f"Initializing camera (index {self.camera_index})...")

        # First try standard backends with V4L2
        for backend, name in backends_to_try:
            print(f"  Trying {name} backend...")
            self.cap = cv2.VideoCapture(self.camera_index, backend)

            if self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                # Test if we can actually read a frame
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                    print(
                        f"  Success! Camera initialized: {width}x{height} @ {fps} FPS ({name})"
                    )
                    return True
                else:
                    self.cap.release()
                    print(f"  {name} backend opened but couldn't read frames")

        # Try libcamera/GStreamer pipelines for Raspberry Pi Camera Module
        print("  Trying libcamera/GStreamer pipelines...")
        for pipeline in libcamera_pipelines:
            try:
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print("  Success with GStreamer pipeline!")
                        return True
                    self.cap.release()
            except Exception as e:
                print(f"  Pipeline failed: {e}")

        print("Error: Could not open camera with any backend")
        print("Troubleshooting tips:")
        print("  1. Check if camera is connected: ls /dev/video*")
        print("  2. For Pi Camera, enable it: sudo raspi-config")
        print("  3. Try: libcamera-hello --list-cameras")
        print("  4. Kill other camera processes: sudo fuser -k /dev/video0")
        return False

    def initialize_filters(self) -> bool:
        """Initialize the filter manager."""
        try:
            print("Loading filters...")
            self.filter_manager = FilterManager()
            if not self.filter_manager.filters:
                print(
                    "Warning: No filters loaded. The app will run but no filters will be displayed."
                )
                print("Please add PNG filter images to assets/filters/ directory.")
            else:
                print(f"Loaded {len(self.filter_manager.filters)} filters")
            return True
        except Exception as e:
            print(f"Error initializing filters: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame: detect face and apply filter.
        Uses frame skipping to reduce CPU load — runs DNN detection
        only every N frames and reuses the last result in between.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (processed frame, face_detected boolean)
        """
        # Flip horizontally for mirror effect
        frame_mirrored = cv2.flip(frame, 1)

        self._frame_count += 1

        # Only run expensive face detection every N frames
        if self._frame_count % self._detect_every_n == 0:
            # Pass BGR directly — no unnecessary color conversion
            results = self.face_mesh.process(frame_mirrored, is_bgr=True)
            self._last_face_detected = bool(results.multi_face_landmarks)
            self._last_face_landmarks = (
                results.multi_face_landmarks[0] if self._last_face_detected else None
            )

        # Get current filter
        current_filter = (
            self.filter_manager.get_current_filter() if self.filter_manager else None
        )

        face_detected = self._last_face_detected

        if self._last_face_landmarks and current_filter:
            face_landmarks = self._last_face_landmarks

            # Get anchor points directly from the face landmarks
            anchor_points = face_landmarks.get_anchor_points()

            # Calculate face width
            face_width = face_landmarks.get_face_width()

            # Overlay filter
            frame_mirrored = overlay_filter(
                frame_mirrored, current_filter, anchor_points, face_width
            )

        return frame_mirrored, face_detected

    def capture_screenshot(self, frame: np.ndarray):
        """Save a screenshot to the desktop."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"magic_mirror_{timestamp}.png"
        filepath = os.path.join(self.screenshots_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            print(f"Screenshot saved: {filepath}")
            self.ui.trigger_screenshot_flash()
        except Exception as e:
            print(f"Error saving screenshot: {e}")

    def toggle_fullscreen(self, window_name: str):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            print("Fullscreen: ON")
        else:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
            )
            print("Fullscreen: OFF")

    def handle_mouse_click(self, event: int, x: int, y: int, flags: int, param: int):
        """Handle mouse clicks for filter navigation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_width = param if param else 640
            
            # Left third of screen - previous filter
            if x < frame_width // 3:
                if self.filter_manager:
                    self.filter_manager.previous_filter()
                    self.ui.trigger_filter_change()
                    print(f"Filter: {self.filter_manager.get_current_filter_name()}")
            # Right third of screen - next filter
            elif x > (frame_width * 2) // 3:
                if self.filter_manager:
                    self.filter_manager.next_filter()
                    self.ui.trigger_filter_change()
                    print(f"Filter: {self.filter_manager.get_current_filter_name()}")

    def run(self):
        """Run the main application loop."""
        # Initialize camera
        if not self.initialize_camera():
            sys.exit(1)

        # Initialize filters
        if not self.initialize_filters():
            print("Warning: Continuing without filters...")

        # Create window
        window_name = "Magic Mirror Face Filter"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if self.fullscreen:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            print("Running in fullscreen mode. Press Q to quit.")
        else:
            print("Running in windowed mode. Press Q to quit.")

        print(
            f"Current filter: {self.filter_manager.get_current_filter_name() if self.filter_manager else 'None'}"
        )

        # Get frame dimensions for mouse callback
        ret, test_frame = self.cap.read()
        if ret:
            frame_width = test_frame.shape[1]
            cv2.setMouseCallback(window_name, self.handle_mouse_click, frame_width)

        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break

                # Process frame
                processed_frame, face_detected = self.process_frame(frame)
                
                frame_h, frame_w = processed_frame.shape[:2]

                # Draw modern UI
                processed_frame = self.ui.draw_controls_hint(processed_frame, frame_w)
                processed_frame = self.ui.draw_face_indicator(processed_frame, face_detected, frame_w)
                processed_frame = self.ui.draw_filter_carousel(
                    processed_frame, self.filter_manager, 
                    self.filter_manager.current_index if self.filter_manager else 0,
                    frame_w, frame_h
                )
                processed_frame = self.ui.draw_screenshot_flash(processed_frame)

                # Display frame
                cv2.imshow(window_name, processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                    print("Quitting...")
                    break
                elif key == ord(" ") or key == 83:  # SPACE or Right Arrow
                    if self.filter_manager:
                        self.filter_manager.next_filter()
                        self.ui.trigger_filter_change()
                        print(f"Filter: {self.filter_manager.get_current_filter_name()}")
                elif key == ord("b") or key == ord("B") or key == 81:  # B or Left Arrow
                    if self.filter_manager:
                        self.filter_manager.previous_filter()
                        self.ui.trigger_filter_change()
                        print(f"Filter: {self.filter_manager.get_current_filter_name()}")
                elif key == ord("c") or key == ord("C"):  # C - Capture screenshot
                    self.capture_screenshot(processed_frame)
                elif key == ord("f") or key == ord("F"):  # F - Toggle fullscreen
                    self.toggle_fullscreen(window_name)
                elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:  # 1-5 direct selection
                    filter_index = key - ord("1")  # Convert to 0-based index
                    if self.filter_manager and self.filter_manager.select_filter(filter_index):
                        self.ui.trigger_filter_change()
                        print(f"Filter: {self.filter_manager.get_current_filter_name()}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.face_mesh:
            self.face_mesh.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Magic Mirror Face Filter - Raspberry Pi Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                    # Run with default camera, fullscreen
  python3 main.py --camera-index 1   # Use camera index 1
  python3 main.py --windowed         # Run in windowed mode (for testing)
        """,
    )

    parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera device index (default: 0)"
    )

    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run in windowed mode instead of fullscreen",
    )

    args = parser.parse_args()

    # Create and run application
    app = MagicMirrorApp(camera_index=args.camera_index, fullscreen=not args.windowed)

    app.run()


if __name__ == "__main__":
    main()

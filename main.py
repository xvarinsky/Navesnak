#!/usr/bin/env python3
"""
Magic Mirror Face Filter - Raspberry Pi Application

A fun, interactive app that shows a live mirrored video with face detection
and overlay filters (mustache, glasses, cat ears, etc.).

Controls:
    SPACE - Next filter
    B     - Previous filter
    Q     - Quit
"""

import cv2
import argparse
import sys
import numpy as np
from face_detector import FaceMeshReplacement
from filters import FilterManager, overlay_filter
from audio_manager import AudioManager


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
        self.audio_manager = AudioManager()

        # Initialize OpenCV DNN Face Detection (ARM64 compatible)
        self.face_mesh = FaceMeshReplacement(min_detection_confidence=0.5)

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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect face and apply filter.

        Args:
            frame: Input frame (BGR)

        Returns:
            Processed frame with filter overlay
        """
        # Flip horizontally for mirror effect
        frame_mirrored = cv2.flip(frame, 1)

        # Convert BGR to RGB for face detection
        frame_rgb = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)

        # Detect faces using OpenCV DNN
        results = self.face_mesh.process(frame_rgb)

        # Get current filter
        current_filter = (
            self.filter_manager.get_current_filter() if self.filter_manager else None
        )

        if results.multi_face_landmarks and current_filter:
            # Process the first detected face
            face_landmarks = results.multi_face_landmarks[0]

            # Get anchor points directly from the face landmarks
            anchor_points = face_landmarks.get_anchor_points()

            # Calculate face width
            face_width = face_landmarks.get_face_width()

            # Overlay filter
            frame_mirrored = overlay_filter(
                frame_mirrored, current_filter, anchor_points, face_width
            )

        return frame_mirrored

    def draw_overlay_text(self, frame: np.ndarray):
        """Draw filter name and instructions on the frame."""
        if self.filter_manager:
            filter_name = self.filter_manager.get_current_filter_name()

            # Draw filter name at top
            cv2.putText(
                frame,
                f"Filter: {filter_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Filter: {filter_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
            )

            # Draw instructions at bottom
            instructions = "SPACE: Next | B: Previous | Q: Quit"
            frame_h = frame.shape[0]
            cv2.putText(
                frame,
                instructions,
                (10, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                instructions,
                (10, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

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

        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Draw overlay text
                self.draw_overlay_text(processed_frame)

                # Display frame
                cv2.imshow(window_name, processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                    print("Quitting...")
                    break
                elif key == ord(" "):  # SPACE
                    if self.filter_manager:
                        self.filter_manager.next_filter()
                        filter_name = self.filter_manager.get_current_filter_name()
                        print(f"Filter: {filter_name}")
                        self.audio_manager.play_for_filter(filter_name)
                elif key == ord("b") or key == ord("B"):  # B
                    if self.filter_manager:
                        self.filter_manager.previous_filter()
                        filter_name = self.filter_manager.get_current_filter_name()
                        print(f"Filter: {filter_name}")
                        self.audio_manager.play_for_filter(filter_name)

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
        if self.audio_manager:
            self.audio_manager.cleanup()
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

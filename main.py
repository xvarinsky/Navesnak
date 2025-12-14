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
import os
import numpy as np
import mediapipe as mp
from filters import FilterManager, compute_anchor_points, overlay_filter


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
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def initialize_camera(self) -> bool:
        """Initialize the camera capture."""
        try:
            print(f"Initializing camera (index {self.camera_index})...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                print("Make sure the camera is connected and not in use by another application.")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual resolution
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            print(f"Camera initialized: {width}x{height} @ {fps} FPS")
            
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def initialize_filters(self) -> bool:
        """Initialize the filter manager."""
        try:
            print("Loading filters...")
            self.filter_manager = FilterManager()
            if not self.filter_manager.filters:
                print("Warning: No filters loaded. The app will run but no filters will be displayed.")
                print("Please add PNG filter images to assets/filters/ directory.")
            else:
                print(f"Loaded {len(self.filter_manager.filters)} filters")
            return True
        except Exception as e:
            print(f"Error initializing filters: {e}")
            return False
    
    def calculate_face_width(self, landmarks, frame_width: int) -> float:
        """
        Calculate face width from landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_width: Width of the frame
        
        Returns:
            Face width in pixels
        """
        # Use left and right cheek points to estimate face width
        left_cheek = landmarks.landmark[234]  # Left cheek
        right_cheek = landmarks.landmark[454]  # Right cheek
        
        face_width = abs(left_cheek.x - right_cheek.x) * frame_width
        return max(face_width, 50.0)  # Minimum width to avoid division issues
    
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
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_mesh.process(frame_rgb)
        
        # Get current filter
        current_filter = self.filter_manager.get_current_filter() if self.filter_manager else None
        
        if results.multi_face_landmarks and current_filter:
            # Process the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate anchor points
            frame_height, frame_width = frame_mirrored.shape[:2]
            anchor_points = compute_anchor_points(face_landmarks, frame_width, frame_height)
            
            # Calculate face width for scaling
            face_width = self.calculate_face_width(face_landmarks, frame_width)
            
            # Overlay filter
            frame_mirrored = overlay_filter(
                frame_mirrored, 
                current_filter, 
                anchor_points, 
                face_width
            )
        
        return frame_mirrored
    
    def draw_overlay_text(self, frame: np.ndarray):
        """Draw filter name and instructions on the frame."""
        if self.filter_manager:
            filter_name = self.filter_manager.get_current_filter_name()
            
            # Draw filter name at top
            cv2.putText(frame, f"Filter: {filter_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Filter: {filter_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Draw instructions at bottom
            instructions = "SPACE: Next | B: Previous | Q: Quit"
            frame_h = frame.shape[0]
            cv2.putText(frame, instructions, (10, frame_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, instructions, (10, frame_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
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
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Running in fullscreen mode. Press Q to quit.")
        else:
            print("Running in windowed mode. Press Q to quit.")
        
        print(f"Current filter: {self.filter_manager.get_current_filter_name() if self.filter_manager else 'None'}")
        
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
                
                if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                    print("Quitting...")
                    break
                elif key == ord(' '):  # SPACE
                    if self.filter_manager:
                        self.filter_manager.next_filter()
                        print(f"Filter: {self.filter_manager.get_current_filter_name()}")
                elif key == ord('b') or key == ord('B'):  # B
                    if self.filter_manager:
                        self.filter_manager.previous_filter()
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
        """
    )
    
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    
    parser.add_argument(
        '--windowed',
        action='store_true',
        help='Run in windowed mode instead of fullscreen'
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = MagicMirrorApp(
        camera_index=args.camera_index,
        fullscreen=not args.windowed
    )
    
    app.run()


if __name__ == "__main__":
    main()


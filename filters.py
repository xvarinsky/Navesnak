"""
Filter configuration and overlay logic for Magic Mirror Face Filter.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class FilterConfig:
    """Configuration for a single filter overlay."""

    def __init__(
        self,
        name: str,
        image_path: str,
        anchor: str,
        scale_factor: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        rotation: float = 0.0,
    ):
        """
        Initialize filter configuration.

        Args:
            name: Display name of the filter
            image_path: Path to PNG image file
            anchor: Anchor point for positioning ('nose', 'eyes_center', 'forehead', 'mouth')
            scale_factor: Scaling factor relative to face width (1.0 = face width)
            offset_x: Horizontal offset in pixels (positive = right)
            offset_y: Vertical offset in pixels (positive = down)
            rotation: Rotation angle in degrees
        """
        self.name = name
        self.image_path = image_path
        self.anchor = anchor
        self.scale_factor = scale_factor
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.rotation = rotation
        self._image = None
        self._image_alpha = None

    def load_image(self) -> bool:
        """Load the filter image with alpha channel, removing white backgrounds."""
        try:
            img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not load filter image: {self.image_path}")
                return False

            # Handle different image formats
            if len(img.shape) == 2:
                # Grayscale image - convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            elif img.shape[2] == 4:
                # Has alpha channel
                alpha = img[:, :, 3]
                img = img[:, :, :3]
            else:
                # No alpha channel - create one based on white background removal
                alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

            # Auto-remove white/near-white background (make it transparent)
            # This helps with generated images that have white backgrounds
            # Convert to grayscale for threshold detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find white/very light pixels (threshold 220-255)
            white_mask = gray > 220

            # Also check if they're actually white (not just bright colors)
            # Check if R, G, B are all similar (grayscale-ish) and high
            b, g, r = cv2.split(img)
            color_diff = np.maximum(
                np.maximum(
                    np.abs(r.astype(int) - g.astype(int)),
                    np.abs(g.astype(int) - b.astype(int)),
                ),
                np.abs(r.astype(int) - b.astype(int)),
            )
            is_grayish = color_diff < 50  # Higher tolerance for color similarity

            # Combine: white AND grayish = background
            background_mask = white_mask & is_grayish

            # Set alpha to 0 where background is detected
            alpha[background_mask] = 0

            # Optional: smooth the alpha edges a bit
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

            self._image = img
            self._image_alpha = alpha

            return True
        except Exception as e:
            print(f"Error loading filter image {self.image_path}: {e}")
            return False

    def get_image(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the loaded image and alpha channel."""
        return self._image, self._image_alpha


class FilterManager:
    """Manages filter configurations and overlay operations."""

    def __init__(self, filters_dir: str = "assets/filters"):
        """
        Initialize filter manager.

        Args:
            filters_dir: Directory containing filter PNG files
        """
        self.filters_dir = filters_dir
        self.filters: List[FilterConfig] = []
        self.current_index = 0
        self._initialize_filters()

    def _initialize_filters(self):
        """Initialize all available filters."""
        import os

        # Define filter configurations
        filter_defs = [
            {
                "name": "Mustache",
                "image": "mustache.png",
                "anchor": "nose",
                "scale_factor": 0.5,
                "offset_y": 25,  # Below nose
            },
            {
                "name": "Glasses",
                "image": "glasses.png",
                "anchor": "eyes_center",
                "scale_factor": 1.0,
                "offset_y": 0,  # On eye level
            },
            {
                "name": "Clown Nose",
                "image": "clown_nose.png",
                "anchor": "nose",
                "scale_factor": 0.5,
                "offset_y": 0,  # On nose tip
            },
            {
                "name": "Unicorn Horn",
                "image": "unicorn.png",
                "anchor": "forehead",
                "scale_factor": 1.4,
                "offset_y": -80,  # Bottom of horn at forehead center
            },
        ]

        for fdef in filter_defs:
            image_path = os.path.join(self.filters_dir, fdef["image"])
            if os.path.exists(image_path):
                config = FilterConfig(
                    name=fdef["name"],
                    image_path=image_path,
                    anchor=fdef["anchor"],
                    scale_factor=fdef.get("scale_factor", 1.0),
                    offset_x=fdef.get("offset_x", 0.0),
                    offset_y=fdef.get("offset_y", 0.0),
                    rotation=fdef.get("rotation", 0.0),
                )
                if config.load_image():
                    self.filters.append(config)
                    print(f"Loaded filter: {fdef['name']}")
            else:
                print(f"Warning: Filter image not found: {image_path}")

        if not self.filters:
            print("Warning: No filters loaded! Please add PNG files to assets/filters/")

    def get_current_filter(self) -> Optional[FilterConfig]:
        """Get the currently active filter."""
        if not self.filters:
            return None
        return self.filters[self.current_index]

    def next_filter(self):
        """Switch to the next filter."""
        if self.filters:
            self.current_index = (self.current_index + 1) % len(self.filters)

    def previous_filter(self):
        """Switch to the previous filter."""
        if self.filters:
            self.current_index = (self.current_index - 1) % len(self.filters)

    def get_current_filter_name(self) -> str:
        """Get the name of the current filter."""
        if not self.filters:
            return "No filters"
        return self.filters[self.current_index].name


def get_landmark_point(landmarks, index: int) -> Tuple[int, int]:
    """Extract a specific landmark point from MediaPipe landmarks."""
    landmark = landmarks.landmark[index]
    return (int(landmark.x), int(landmark.y))


def compute_anchor_points(
    landmarks, frame_width: int, frame_height: int
) -> Dict[str, Tuple[int, int]]:
    """
    Compute anchor points from face landmarks.

    NOTE: This function is kept for backwards compatibility.
    When using the new OpenCV DNN face detector, anchor points are computed
    directly by the FaceLandmarks.get_anchor_points() method.

    Args:
        landmarks: Face landmarks object (MediaPipe format - legacy)
        frame_width: Width of the frame
        frame_height: Height of the frame

    Returns:
        Dictionary of anchor point names to (x, y) coordinates
    """
    # This function is a legacy wrapper for MediaPipe landmarks
    # For the new OpenCV DNN detector, use FaceLandmarks.get_anchor_points()

    def get_point(idx):
        lm = landmarks.landmark[idx]
        return (int(lm.x * frame_width), int(lm.y * frame_height))

    # Get key landmarks (MediaPipe indices)
    nose_tip = get_point(1)
    left_eye = get_point(33)
    right_eye = get_point(263)
    forehead = get_point(10)
    mouth_center = get_point(13)

    # Compute derived points
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    return {
        "nose": nose_tip,
        "eyes_center": eyes_center,
        "forehead": forehead,
        "mouth": mouth_center,
        "left_eye": left_eye,
        "right_eye": right_eye,
    }


def overlay_filter(
    frame: np.ndarray,
    filter_config: FilterConfig,
    anchor_points: Dict[str, Tuple[int, int]],
    face_width: float,
) -> np.ndarray:
    """
    Overlay a filter image onto the frame at the specified anchor point.

    Args:
        frame: Input frame (BGR)
        filter_config: Filter configuration
        anchor_points: Dictionary of anchor point coordinates
        face_width: Width of detected face (for scaling)

    Returns:
        Frame with filter overlaid
    """
    if filter_config is None:
        return frame

    # Get anchor point
    anchor = anchor_points.get(filter_config.anchor)
    if anchor is None:
        return frame

    # Get filter image
    filter_img, filter_alpha = filter_config.get_image()
    if filter_img is None or filter_alpha is None:
        return frame

    # Calculate scale based on face width
    target_width = int(face_width * filter_config.scale_factor)
    if target_width <= 0:
        return frame

    # Resize filter image maintaining aspect ratio
    aspect_ratio = filter_img.shape[1] / filter_img.shape[0]
    target_height = int(target_width / aspect_ratio)

    if target_height <= 0 or target_width <= 0:
        return frame

    filter_resized = cv2.resize(filter_img, (target_width, target_height))
    alpha_resized = cv2.resize(filter_alpha, (target_width, target_height))

    # Normalize alpha to 0-1 range
    alpha_normalized = alpha_resized.astype(np.float32) / 255.0

    # Calculate position (center the filter on anchor point)
    x_offset = int(filter_config.offset_x)
    y_offset = int(filter_config.offset_y)

    x1 = anchor[0] - target_width // 2 + x_offset
    y1 = anchor[1] - target_height // 2 + y_offset
    x2 = x1 + target_width
    y2 = y1 + target_height

    # Check bounds
    frame_h, frame_w = frame.shape[:2]

    # Calculate crop if filter goes outside frame
    crop_x1 = max(0, -x1)
    crop_y1 = max(0, -y1)
    crop_x2 = min(target_width, frame_w - x1)
    crop_y2 = min(target_height, frame_h - y1)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return frame

    # Adjust coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    # Extract region of interest
    roi = frame[y1:y2, x1:x2]

    # Crop filter to fit
    filter_cropped = filter_resized[crop_y1:crop_y2, crop_x1:crop_x2]
    alpha_cropped = alpha_normalized[crop_y1:crop_y2, crop_x1:crop_x2]

    # Blend filter with frame using alpha
    if len(roi.shape) == 3 and len(filter_cropped.shape) == 3:
        for c in range(3):
            roi[:, :, c] = (
                alpha_cropped * filter_cropped[:, :, c]
                + (1 - alpha_cropped) * roi[:, :, c]
            )

    return frame

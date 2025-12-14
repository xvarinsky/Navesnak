#!/usr/bin/env python3
"""
Face detection and landmark detection using OpenCV DNN.

This module provides ARM64-compatible face detection using:
- OpenCV DNN SSD face detector (Caffe model)
- OpenCV FacemarkLBF for facial landmarks

Models are automatically downloaded on first use.
"""

import cv2
import numpy as np
import os
import urllib.request
from typing import Dict, List, Tuple, Optional


# Model URLs
FACE_DETECTOR_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_DETECTOR_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
LANDMARK_MODEL_URL = (
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
)

# Default paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_file(url: str, destination: str, description: str = "file") -> bool:
    """
    Download a file from URL to destination.

    Args:
        url: URL to download from
        destination: Local file path to save to
        description: Description for progress messages

    Returns:
        True if successful, False otherwise
    """
    if os.path.exists(destination):
        return True

    try:
        print(f"Downloading {description}...")
        print(f"  URL: {url}")
        print(f"  Destination: {destination}")

        # Create directory if needed
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Download with progress
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                print(f"\r  Progress: {min(percent, 100)}%", end="", flush=True)

        urllib.request.urlretrieve(url, destination, reporthook)
        print()  # New line after progress
        print("  Downloaded successfully!")
        return True
    except Exception as e:
        print(f"  Error downloading {description}: {e}")
        return False


def ensure_models_exist() -> Tuple[str, str, str]:
    """
    Ensure all required models exist, downloading if necessary.

    Returns:
        Tuple of (prototxt_path, caffemodel_path, landmark_model_path)

    Raises:
        RuntimeError: If models cannot be downloaded
    """
    prototxt_path = os.path.join(MODELS_DIR, "deploy.prototxt")
    caffemodel_path = os.path.join(
        MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel"
    )
    landmark_path = os.path.join(MODELS_DIR, "lbfmodel.yaml")

    # Download face detector prototxt
    if not download_file(
        FACE_DETECTOR_PROTOTXT_URL, prototxt_path, "face detector architecture"
    ):
        raise RuntimeError("Failed to download face detector prototxt")

    # Download face detector model
    if not download_file(
        FACE_DETECTOR_MODEL_URL, caffemodel_path, "face detector model"
    ):
        raise RuntimeError("Failed to download face detector model")

    # Download landmark model
    if not download_file(LANDMARK_MODEL_URL, landmark_path, "landmark model"):
        raise RuntimeError("Failed to download landmark model")

    return prototxt_path, caffemodel_path, landmark_path


class FaceDetector:
    """OpenCV DNN-based face detector using SSD ResNet."""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize face detector.

        Args:
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.net = None
        self._load_model()

    def _load_model(self):
        """Load the face detection model."""
        prototxt_path, caffemodel_path, _ = ensure_models_exist()

        print("Loading face detection model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # Try to use OpenCL if available (helps on some ARM devices)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Face detection model loaded.")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame.

        Args:
            frame: Input BGR frame

        Returns:
            List of face bounding boxes as (x, y, w, h) tuples
        """
        if self.net is None:
            return []

        h, w = frame.shape[:2]

        # Prepare image for DNN
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Ensure valid bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces


class LandmarkDetector:
    """OpenCV FacemarkLBF-based facial landmark detector."""

    # 68-point landmark indices for key facial features
    # Based on iBUG 68-point model
    LANDMARK_INDICES = {
        "nose_tip": 30,
        "nose_bridge": 27,
        "left_eye_outer": 36,
        "left_eye_inner": 39,
        "right_eye_inner": 42,
        "right_eye_outer": 45,
        "left_eyebrow_outer": 17,
        "right_eyebrow_outer": 26,
        "mouth_left": 48,
        "mouth_right": 54,
        "mouth_top": 51,
        "mouth_bottom": 57,
        "jaw_left": 0,
        "jaw_right": 16,
        "chin": 8,
    }

    def __init__(self):
        """Initialize landmark detector."""
        self.facemark = None
        self._load_model()

    def _load_model(self):
        """Load the landmark detection model."""
        _, _, landmark_path = ensure_models_exist()

        print("Loading landmark detection model...")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(landmark_path)
        print("Landmark detection model loaded.")

    def detect_landmarks(
        self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Detect facial landmarks for a face.

        Args:
            frame: Input BGR frame
            face_rect: Face bounding box as (x, y, w, h)

        Returns:
            Array of 68 landmark points as (x, y) coordinates, or None if detection fails
        """
        if self.facemark is None:
            return None

        # Convert to format expected by facemark
        faces = np.array([[face_rect[0], face_rect[1], face_rect[2], face_rect[3]]])

        try:
            # Detect landmarks
            success, landmarks = self.facemark.fit(frame, faces)

            if success and len(landmarks) > 0:
                return landmarks[0][0]  # Return first face's landmarks
        except Exception as e:
            print(f"Landmark detection error: {e}")

        return None

    def get_anchor_points(self, landmarks: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Compute anchor points from 68-point landmarks.

        Args:
            landmarks: Array of 68 landmark points

        Returns:
            Dictionary of anchor point names to (x, y) coordinates
        """
        if landmarks is None or len(landmarks) < 68:
            return {}

        def get_point(idx: int) -> Tuple[int, int]:
            return (int(landmarks[idx][0]), int(landmarks[idx][1]))

        def get_midpoint(idx1: int, idx2: int) -> Tuple[int, int]:
            p1 = landmarks[idx1]
            p2 = landmarks[idx2]
            return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

        # Compute key anchor points
        nose_tip = get_point(30)

        # Eyes center (between inner corners)
        left_eye_center = get_midpoint(36, 39)
        right_eye_center = get_midpoint(42, 45)
        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) // 2,
            (left_eye_center[1] + right_eye_center[1]) // 2,
        )

        # Forehead (estimate from nose bridge, above eyebrows)
        nose_bridge = get_point(27)
        forehead = (
            nose_bridge[0],
            nose_bridge[1] - int(abs(nose_bridge[1] - eyes_center[1]) * 0.8),
        )

        # Mouth center
        mouth_center = get_midpoint(51, 57)

        return {
            "nose": nose_tip,
            "eyes_center": eyes_center,
            "forehead": forehead,
            "mouth": mouth_center,
            "left_eye": left_eye_center,
            "right_eye": right_eye_center,
        }

    def calculate_face_width(self, landmarks: np.ndarray) -> float:
        """
        Calculate face width from landmarks.

        Args:
            landmarks: Array of 68 landmark points

        Returns:
            Face width in pixels
        """
        if landmarks is None or len(landmarks) < 68:
            return 100.0  # Default value

        # Use jaw points to estimate face width
        jaw_left = landmarks[0]
        jaw_right = landmarks[16]

        face_width = np.sqrt(
            (jaw_right[0] - jaw_left[0]) ** 2 + (jaw_right[1] - jaw_left[1]) ** 2
        )

        return max(face_width, 50.0)  # Minimum width


class FaceMeshReplacement:
    """
    Drop-in replacement for MediaPipe FaceMesh using OpenCV DNN.

    This class provides a similar interface to MediaPipe FaceMesh
    for easier migration.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the face mesh replacement.

        Args:
            min_detection_confidence: Minimum confidence for face detection
        """
        self.face_detector = FaceDetector(confidence_threshold=min_detection_confidence)
        self.landmark_detector = LandmarkDetector()

    def process(self, frame_rgb: np.ndarray) -> "FaceResults":
        """
        Process a frame and detect faces with landmarks.

        Args:
            frame_rgb: Input RGB frame

        Returns:
            FaceResults object with detected faces
        """
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = self.face_detector.detect_faces(frame_bgr)

        results = FaceResults()

        for face_rect in faces:
            landmarks = self.landmark_detector.detect_landmarks(frame_bgr, face_rect)
            if landmarks is not None:
                face_data = FaceLandmarks(
                    landmarks=landmarks,
                    face_rect=face_rect,
                    landmark_detector=self.landmark_detector,
                )
                results.add_face(face_data)

        return results

    def close(self):
        """Clean up resources."""
        pass  # No cleanup needed for OpenCV models


class FaceResults:
    """Container for face detection results."""

    def __init__(self):
        self.multi_face_landmarks: List["FaceLandmarks"] = []

    def add_face(self, face_landmarks: "FaceLandmarks"):
        self.multi_face_landmarks.append(face_landmarks)


class FaceLandmarks:
    """
    Container for face landmarks.

    Provides access to landmarks in a format compatible with the existing code.
    """

    def __init__(
        self,
        landmarks: np.ndarray,
        face_rect: Tuple[int, int, int, int],
        landmark_detector: LandmarkDetector,
    ):
        self._landmarks = landmarks
        self._face_rect = face_rect
        self._landmark_detector = landmark_detector
        self._anchor_points: Optional[Dict[str, Tuple[int, int]]] = None
        self._face_width: Optional[float] = None

    @property
    def raw_landmarks(self) -> np.ndarray:
        """Get raw 68-point landmarks."""
        return self._landmarks

    @property
    def face_rect(self) -> Tuple[int, int, int, int]:
        """Get face bounding box."""
        return self._face_rect

    def get_anchor_points(self) -> Dict[str, Tuple[int, int]]:
        """Get computed anchor points."""
        if self._anchor_points is None:
            self._anchor_points = self._landmark_detector.get_anchor_points(
                self._landmarks
            )
        return self._anchor_points

    def get_face_width(self) -> float:
        """Get calculated face width."""
        if self._face_width is None:
            self._face_width = self._landmark_detector.calculate_face_width(
                self._landmarks
            )
        return self._face_width

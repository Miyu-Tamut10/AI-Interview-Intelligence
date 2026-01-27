"""
Facial Engagement Analysis Module
Analyzes facial cues, eye contact, and engagement from video frames
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import glob

from .config import MODELS_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MediaPipe (graceful fallback)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Install with: pip install mediapipe")


@dataclass
class FacialMetrics:
    """Data class for facial analysis metrics"""
    eye_contact_ratio: float
    head_stability_score: float
    facial_expression_variance: float
    engagement_score: float
    frames_analyzed: int
    frames_with_face: int
    face_detection_rate: float
    avg_face_confidence: float


class FaceAnalyzer:
    """
    Analyzes facial features and engagement from video frames
    """
    
    def __init__(self, frames_dir: str):
        """
        Initialize face analyzer
        
        Args:
            frames_dir: Directory containing extracted video frames
        """
        self.frames_dir = Path(frames_dir)
        
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
        
        # Get all frame files
        self.frame_files = sorted(glob.glob(str(self.frames_dir / "*.jpg")))
        
        if not self.frame_files:
            raise ValueError(f"No frames found in {self.frames_dir}")
        
        logger.info(f"Found {len(self.frame_files)} frames to analyze")
        
        # Initialize MediaPipe Face Mesh
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=MODELS_CONFIG["facial"]["max_faces"],
                min_detection_confidence=MODELS_CONFIG["facial"]["face_detection_confidence"]
            )
            logger.info("✓ MediaPipe Face Mesh initialized")
        else:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✓ Using OpenCV Haar Cascade (fallback)")
    
    def detect_face_mediapipe(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect face and extract features using MediaPipe
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Dictionary with face landmarks and metrics, or None if no face
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face (assuming single person interview)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract key landmarks
        h, w = image.shape[:2]
        landmarks_2d = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append((x, y))
        
        # Extract specific features
        # Left eye: landmarks 33, 133
        # Right eye: landmarks 362, 263
        # Nose tip: landmark 1
        # Chin: landmark 152
        
        left_eye = np.array([landmarks_2d[33], landmarks_2d[133]])
        right_eye = np.array([landmarks_2d[362], landmarks_2d[263]])
        nose_tip = np.array(landmarks_2d[1])
        chin = np.array(landmarks_2d[152])
        
        return {
            "landmarks": landmarks_2d,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "nose_tip": nose_tip,
            "chin": chin,
            "confidence": 1.0  # MediaPipe doesn't provide confidence
        }
    
    def detect_face_opencv(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect face using OpenCV Haar Cascade (fallback)
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Dictionary with face region, or None if no face
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(faces) == 0:
            return None
        
        # Get first face
        x, y, w, h = faces[0]
        
        # Estimate key points from bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        return {
            "bbox": (x, y, w, h),
            "center": (center_x, center_y),
            "confidence": 0.8  # Arbitrary for Haar Cascade
        }
    
    def analyze_eye_contact(self, face_data: Dict[str, Any], image_shape: Tuple) -> float:
        """
        Analyze eye contact based on gaze direction
        
        Args:
            face_data: Face detection data
            image_shape: Image dimensions (height, width)
            
        Returns:
            Eye contact score (0-1)
        """
        if not self.use_mediapipe or "left_eye" not in face_data:
            # Fallback: assume center position is good eye contact
            if "center" in face_data:
                h, w = image_shape[:2]
                center_x, center_y = face_data["center"]
                
                # Check if face is centered
                x_centered = abs(center_x - w/2) < w * 0.2
                y_centered = abs(center_y - h/2) < h * 0.2
                
                if x_centered and y_centered:
                    return 0.8
                else:
                    return 0.5
            return 0.5
        
        # With MediaPipe, analyze eye position
        h, w = image_shape[:2]
        
        # Check if eyes are in center region (looking at camera)
        left_eye_center = np.mean(face_data["left_eye"], axis=0)
        right_eye_center = np.mean(face_data["right_eye"], axis=0)
        
        eyes_center = (left_eye_center + right_eye_center) / 2
        
        # Normalized distance from image center
        center_x, center_y = w / 2, h / 2
        distance = np.linalg.norm(eyes_center - np.array([center_x, center_y]))
        max_distance = np.linalg.norm([w/2, h/2])
        
        normalized_distance = distance / max_distance
        
        # Convert distance to score (closer to center = better)
        eye_contact_score = max(0, 1 - normalized_distance)
        
        return eye_contact_score
    
    def analyze_head_stability(self, face_positions: List[Tuple[int, int]]) -> float:
        """
        Analyze head movement stability across frames
        
        Args:
            face_positions: List of (x, y) face center positions
            
        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(face_positions) < 2:
            return 1.0
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(1, len(face_positions)):
            prev_pos = np.array(face_positions[i-1])
            curr_pos = np.array(face_positions[i])
            movement = np.linalg.norm(curr_pos - prev_pos)
            movements.append(movement)
        
        # Calculate average movement
        avg_movement = np.mean(movements)
        
        # Normalize (assuming image width ~640px, movement > 50px is significant)
        normalized_movement = min(avg_movement / 50.0, 1.0)
        
        # Convert to stability score
        stability = 1 - normalized_movement
        
        return max(0, stability)
    
    def analyze_expression_variance(self, face_data_list: List[Dict]) -> float:
        """
        Analyze facial expression variance (indicator of engagement/stress)
        
        Args:
            face_data_list: List of face detection data across frames
            
        Returns:
            Expression variance score (0-1)
        """
        if len(face_data_list) < 5:
            return 0.5  # Not enough data
        
        # For simplicity, use face position variance as proxy
        # In production, would analyze actual expression landmarks
        
        if self.use_mediapipe and "nose_tip" in face_data_list[0]:
            nose_positions = [data["nose_tip"] for data in face_data_list]
            nose_y_coords = [pos[1] for pos in nose_positions]
            
            # Calculate variance
            variance = np.var(nose_y_coords)
            
            # Normalize (small variance is good, means stable expression)
            # But too rigid is also not natural
            normalized_variance = min(variance / 100.0, 1.0)
            
            # Optimal variance is moderate (0.2-0.4)
            if 0.2 <= normalized_variance <= 0.4:
                return 0.9
            elif normalized_variance < 0.2:
                return 0.6  # Too rigid
            else:
                return max(0.3, 1 - normalized_variance)
        
        return 0.5  # Default moderate score
    
    def analyze_frames(self, max_frames: Optional[int] = None) -> FacialMetrics:
        """
        Analyze all frames for facial engagement
        
        Args:
            max_frames: Maximum number of frames to analyze (None = all)
            
        Returns:
            FacialMetrics object with analysis results
        """
        logger.info("=" * 60)
        logger.info("Starting facial analysis")
        logger.info("=" * 60)
        
        frames_to_analyze = self.frame_files
        if max_frames:
            # Sample frames uniformly
            step = max(1, len(self.frame_files) // max_frames)
            frames_to_analyze = self.frame_files[::step][:max_frames]
        
        face_data_list = []
        eye_contact_scores = []
        face_positions = []
        confidences = []
        
        frames_analyzed = 0
        frames_with_face = 0
        
        for frame_path in frames_to_analyze:
            # Load frame
            image = cv2.imread(frame_path)
            if image is None:
                continue
            
            frames_analyzed += 1
            
            # Detect face
            if self.use_mediapipe:
                face_data = self.detect_face_mediapipe(image)
            else:
                face_data = self.detect_face_opencv(image)
            
            if face_data is None:
                continue
            
            frames_with_face += 1
            face_data_list.append(face_data)
            
            # Extract face position
            if "center" in face_data:
                face_positions.append(face_data["center"])
            elif "nose_tip" in face_data:
                face_positions.append(tuple(face_data["nose_tip"]))
            
            # Analyze eye contact for this frame
            eye_score = self.analyze_eye_contact(face_data, image.shape)
            eye_contact_scores.append(eye_score)
            
            # Store confidence
            confidences.append(face_data.get("confidence", 0.8))
        
        logger.info(f"Analyzed {frames_analyzed} frames, detected face in {frames_with_face}")
        
        # Compute aggregate metrics
        face_detection_rate = frames_with_face / frames_analyzed if frames_analyzed > 0 else 0
        
        eye_contact_ratio = np.mean(eye_contact_scores) if eye_contact_scores else 0.0
        head_stability = self.analyze_head_stability(face_positions)
        expression_variance = self.analyze_expression_variance(face_data_list)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate overall engagement score
        engagement_score = (
            0.5 * eye_contact_ratio +
            0.3 * head_stability +
            0.2 * face_detection_rate
        )
        
        metrics = FacialMetrics(
            eye_contact_ratio=round(float(eye_contact_ratio), 3),
            head_stability_score=round(float(head_stability), 3),
            facial_expression_variance=round(float(expression_variance), 3),
            engagement_score=round(float(engagement_score), 3),
            frames_analyzed=frames_analyzed,
            frames_with_face=frames_with_face,
            face_detection_rate=round(face_detection_rate, 3),
            avg_face_confidence=round(float(avg_confidence), 3)
        )
        
        logger.info("=" * 60)
        logger.info("✓ Facial analysis completed")
        logger.info(f"Engagement Score: {metrics.engagement_score}")
        logger.info(f"Eye Contact Ratio: {metrics.eye_contact_ratio}")
        logger.info("=" * 60)
        
        return metrics
    
    def cleanup(self):
        """Release resources"""
        if self.use_mediapipe:
            self.face_mesh.close()
        logger.info("Face analyzer cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def analyze_facial_engagement(frames_dir: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze facial engagement
    
    Args:
        frames_dir: Directory containing video frames
        max_frames: Maximum frames to analyze
        
    Returns:
        Analysis results as dictionary
    """
    with FaceAnalyzer(frames_dir) as analyzer:
        metrics = analyzer.analyze_frames(max_frames)
        return asdict(metrics)


if __name__ == "__main__":
    print("Face Analysis Module - Ready")
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")

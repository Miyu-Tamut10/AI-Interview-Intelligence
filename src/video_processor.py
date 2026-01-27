"""
Video Processing Module
Handles video upload, validation, and extraction of audio + frames
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import subprocess
import json

from .config import VIDEO_CONFIG, OUTPUT_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes interview video files to extract audio and frames
    for multimodal analysis
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        self.validate_video()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"Duration: {self.duration:.2f}s, FPS: {self.fps:.2f}, Frames: {self.frame_count}")
    
    def validate_video(self) -> bool:
        """Validate video file exists and has supported format"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        if self.video_path.suffix.lower() not in VIDEO_CONFIG["supported_formats"]:
            raise ValueError(
                f"Unsupported video format: {self.video_path.suffix}. "
                f"Supported: {VIDEO_CONFIG['supported_formats']}"
            )
        
        return True
    
    def get_video_metadata(self) -> Dict[str, Any]:
        """
        Extract video metadata
        
        Returns:
            Dictionary containing video metadata
        """
        return {
            "filename": self.video_path.name,
            "duration_seconds": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "resolution": (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "codec": int(self.cap.get(cv2.CAP_PROP_FOURCC))
        }
    
    def extract_audio(self, output_path: Optional[str] = None) -> Path:
        """
        Extract audio from video using FFmpeg
        
        Args:
            output_path: Output path for audio file (default: auto-generated)
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = OUTPUT_DIR / f"{self.video_path.stem}_audio.wav"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Extracting audio to: {output_path}")
        
        # Use FFmpeg to extract audio
        command = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            logger.info(f"✓ Audio extracted successfully: {output_path.name}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract audio: {e}")
    
    def extract_frames(
        self,
        output_dir: Optional[str] = None,
        fps: Optional[float] = None
    ) -> Tuple[Path, int]:
        """
        Extract frames from video for facial analysis
        
        Args:
            output_dir: Directory to save frames (default: auto-generated)
            fps: Frames per second to extract (default: from config)
            
        Returns:
            Tuple of (output_directory_path, number_of_frames_extracted)
        """
        if fps is None:
            fps = VIDEO_CONFIG["frame_extraction_fps"]
        
        if output_dir is None:
            output_dir = OUTPUT_DIR / f"{self.video_path.stem}_frames"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate frame interval
        frame_interval = int(self.fps / fps)
        
        logger.info(f"Extracting frames at {fps} FPS to: {output_dir}")
        
        frame_idx = 0
        extracted_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_idx % frame_interval == 0:
                frame_path = output_dir / f"frame_{extracted_count:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
            
            frame_idx += 1
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        logger.info(f"✓ Extracted {extracted_count} frames")
        return output_dir, extracted_count
    
    def extract_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a specific frame at given timestamp
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array or None if failed
        """
        frame_number = int(timestamp * self.fps)
        
        if frame_number >= self.frame_count:
            logger.warning(f"Timestamp {timestamp}s exceeds video duration")
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            return frame
        return None
    
    def process_video(self) -> Dict[str, Any]:
        """
        Complete video processing pipeline
        Extracts both audio and frames
        
        Returns:
            Dictionary containing paths and metadata
        """
        logger.info("=" * 60)
        logger.info("Starting video processing pipeline")
        logger.info("=" * 60)
        
        # Get metadata
        metadata = self.get_video_metadata()
        
        # Extract audio
        audio_path = self.extract_audio()
        
        # Extract frames
        frames_dir, frame_count = self.extract_frames()
        
        result = {
            "status": "success",
            "metadata": metadata,
            "audio_path": str(audio_path),
            "frames_directory": str(frames_dir),
            "frames_extracted": frame_count,
            "processing_timestamp": None  # Will be set by caller
        }
        
        logger.info("=" * 60)
        logger.info("✓ Video processing completed successfully")
        logger.info("=" * 60)
        
        return result
    
    def cleanup(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
        logger.info("Video processor cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def process_interview_video(video_path: str) -> Dict[str, Any]:
    """
    Convenience function to process interview video
    
    Args:
        video_path: Path to video file
        
    Returns:
        Processing results dictionary
    """
    with VideoProcessor(video_path) as processor:
        return processor.process_video()


if __name__ == "__main__":
    # Test the module
    print("Video Processor Module - Ready")
    print(f"Supported formats: {VIDEO_CONFIG['supported_formats']}")

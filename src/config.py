"""
Configuration management for the Interview Intelligence System
Centralized configuration for all modules
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = DATA_DIR / "models_cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS_CONFIG = {
    "whisper": {
        "model_name": "base",  # Options: tiny, base, small, medium, large
        "language": "en",
        "task": "transcribe"
    },
    "nlp": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": 0.6,
        "max_length": 512
    },
    "facial": {
        "face_detection_confidence": 0.5,
        "tracking_confidence": 0.5,
        "max_faces": 1
    }
}

# Audio analysis configuration
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "hop_length": 512,
    "n_mfcc": 13,
    "filler_words": ["um", "uh", "like", "you know", "basically", "actually", "literally"],
    "ideal_speech_rate": (120, 160),  # words per minute
    "max_pause_duration": 3.0  # seconds
}

# Scoring weights (must sum to 1.0)
SCORING_WEIGHTS = {
    "nlp_score": 0.35,
    "speech_score": 0.30,
    "facial_score": 0.20,
    "structure_score": 0.15
}

# Thresholds
THRESHOLDS = {
    "excellent": 0.85,
    "good": 0.70,
    "average": 0.55,
    "needs_improvement": 0.40
}

# Video processing configuration
VIDEO_CONFIG = {
    "frame_extraction_fps": 1,  # Extract 1 frame per second
    "max_video_duration": 600,  # 10 minutes max
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "system.log"
}


def get_config(module: str) -> Dict[str, Any]:
    """
    Get configuration for a specific module
    
    Args:
        module: Module name (whisper, nlp, facial, audio, video)
        
    Returns:
        Configuration dictionary
    """
    config_map = {
        "whisper": MODELS_CONFIG["whisper"],
        "nlp": MODELS_CONFIG["nlp"],
        "facial": MODELS_CONFIG["facial"],
        "audio": AUDIO_CONFIG,
        "video": VIDEO_CONFIG,
        "scoring": SCORING_WEIGHTS
    }
    
    return config_map.get(module, {})


def validate_config():
    """Validate that all configuration values are correct"""
    # Check scoring weights sum to 1.0
    weight_sum = sum(SCORING_WEIGHTS.values())
    assert abs(weight_sum - 1.0) < 0.01, f"Scoring weights must sum to 1.0, got {weight_sum}"
    
    # Check directories exist
    assert PROJECT_ROOT.exists(), "Project root directory not found"
    
    return True


if __name__ == "__main__":
    validate_config()
    print("✓ Configuration validated successfully")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Scoring Weights: {SCORING_WEIGHTS}")

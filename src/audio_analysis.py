"""
Audio Analysis Module
Analyzes speech patterns, fluency, confidence, and vocal characteristics
"""

import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re
from dataclasses import dataclass, asdict

from .config import AUDIO_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpeechMetrics:
    """Data class for speech analysis metrics"""
    speech_rate: float  # Words per minute
    pause_count: int
    avg_pause_duration: float
    max_pause_duration: float
    filler_words_count: int
    filler_ratio: float
    total_words: int
    speaking_duration: float
    confidence_score: float
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float


class AudioAnalyzer:
    """
    Analyzes audio features to evaluate speech quality and confidence
    """
    
    def __init__(self, audio_path: str, transcript: str = ""):
        """
        Initialize audio analyzer
        
        Args:
            audio_path: Path to audio file (.wav)
            transcript: Transcribed text (optional, needed for full analysis)
        """
        self.audio_path = Path(audio_path)
        self.transcript = transcript
        
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        
        # Load audio
        self.y, self.sr = librosa.load(
            str(self.audio_path),
            sr=AUDIO_CONFIG["sample_rate"]
        )
        
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        logger.info(f"Audio loaded: {self.audio_path.name}")
        logger.info(f"Duration: {self.duration:.2f}s, Sample rate: {self.sr}Hz")
    
    def extract_acoustic_features(self) -> Dict[str, Any]:
        """
        Extract acoustic features from audio signal
        
        Returns:
            Dictionary of acoustic features
        """
        logger.info("Extracting acoustic features...")
        
        # Extract pitch (F0) using librosa
        pitches, magnitudes = librosa.piptrack(
            y=self.y,
            sr=self.sr,
            hop_length=AUDIO_CONFIG["hop_length"]
        )
        
        # Get fundamental frequency
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Only non-zero pitches
                pitch_values.append(pitch)
        
        pitch_values = np.array(pitch_values)
        
        # Extract MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(
            y=self.y,
            sr=self.sr,
            n_mfcc=AUDIO_CONFIG["n_mfcc"]
        )
        
        # Extract energy/amplitude
        rms = librosa.feature.rms(y=self.y)[0]
        
        # Extract zero crossing rate (voice vs silence indicator)
        zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        
        features = {
            "pitch_mean": float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0,
            "pitch_std": float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0,
            "pitch_min": float(np.min(pitch_values)) if len(pitch_values) > 0 else 0.0,
            "pitch_max": float(np.max(pitch_values)) if len(pitch_values) > 0 else 0.0,
            "energy_mean": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
            "mfcc_mean": float(np.mean(mfcc)),
            "mfcc_std": float(np.std(mfcc)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr))
        }
        
        logger.info("✓ Acoustic features extracted")
        return features
    
    def detect_pauses(self, threshold_db: float = -40) -> List[Tuple[float, float]]:
        """
        Detect pauses in speech
        
        Args:
            threshold_db: Energy threshold in dB for silence detection
            
        Returns:
            List of (start_time, end_time) tuples for pauses
        """
        # Get RMS energy
        rms = librosa.feature.rms(y=self.y)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms)
        
        # Detect silence frames
        silence_frames = rms_db < threshold_db
        
        # Convert frames to time
        times = librosa.frames_to_time(
            np.arange(len(silence_frames)),
            sr=self.sr,
            hop_length=AUDIO_CONFIG["hop_length"]
        )
        
        # Find continuous silence segments
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_pause:
                pause_start = times[i]
                in_pause = True
            elif not is_silent and in_pause:
                pause_end = times[i]
                pause_duration = pause_end - pause_start
                
                # Only count significant pauses (> 0.5 seconds)
                if pause_duration > 0.5:
                    pauses.append((pause_start, pause_end))
                
                in_pause = False
        
        return pauses
    
    def analyze_filler_words(self) -> Dict[str, Any]:
        """
        Analyze usage of filler words in transcript
        
        Returns:
            Dictionary with filler word analysis
        """
        if not self.transcript:
            logger.warning("No transcript provided for filler word analysis")
            return {
                "filler_words_count": 0,
                "filler_ratio": 0.0,
                "filler_breakdown": {}
            }
        
        # Normalize transcript
        text_lower = self.transcript.lower()
        
        # Count filler words
        filler_breakdown = {}
        total_fillers = 0
        
        for filler in AUDIO_CONFIG["filler_words"]:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(filler) + r'\b'
            count = len(re.findall(pattern, text_lower))
            
            if count > 0:
                filler_breakdown[filler] = count
                total_fillers += count
        
        # Count total words
        words = text_lower.split()
        total_words = len(words)
        
        filler_ratio = total_fillers / total_words if total_words > 0 else 0.0
        
        return {
            "filler_words_count": total_fillers,
            "filler_ratio": round(filler_ratio, 4),
            "filler_breakdown": filler_breakdown,
            "total_words": total_words
        }
    
    def calculate_speech_rate(self) -> float:
        """
        Calculate speaking rate (words per minute)
        
        Returns:
            Speech rate in WPM
        """
        if not self.transcript:
            logger.warning("No transcript provided for speech rate calculation")
            return 0.0
        
        words = self.transcript.split()
        word_count = len(words)
        
        # Duration in minutes
        duration_minutes = self.duration / 60.0
        
        if duration_minutes == 0:
            return 0.0
        
        speech_rate = word_count / duration_minutes
        
        return round(speech_rate, 2)
    
    def calculate_confidence_score(self, acoustic_features: Dict, pauses: List) -> float:
        """
        Calculate confidence score based on multiple factors
        
        Factors:
        - Pitch stability (lower std = more confident)
        - Energy level (moderate energy = confident)
        - Pause patterns (excessive pauses = less confident)
        - Speech rate (optimal range = confident)
        
        Args:
            acoustic_features: Dictionary of acoustic features
            pauses: List of pause segments
            
        Returns:
            Confidence score (0-1)
        """
        scores = []
        
        # 1. Pitch stability score (inverse of coefficient of variation)
        pitch_mean = acoustic_features["pitch_mean"]
        pitch_std = acoustic_features["pitch_std"]
        
        if pitch_mean > 0:
            pitch_cv = pitch_std / pitch_mean
            # Lower CV = more stable = higher score
            pitch_score = max(0, 1 - (pitch_cv / 0.5))  # Normalize around 0.5 CV
            scores.append(pitch_score)
        
        # 2. Energy score (moderate energy is good)
        energy_mean = acoustic_features["energy_mean"]
        # Optimal range: 0.01 to 0.1
        if energy_mean < 0.01:
            energy_score = energy_mean / 0.01
        elif energy_mean > 0.1:
            energy_score = max(0, 1 - (energy_mean - 0.1) / 0.1)
        else:
            energy_score = 1.0
        scores.append(energy_score)
        
        # 3. Pause score
        avg_pause_duration = np.mean([end - start for start, end in pauses]) if pauses else 0
        max_pause_allowed = AUDIO_CONFIG["max_pause_duration"]
        
        if avg_pause_duration <= 1.0:
            pause_score = 1.0
        elif avg_pause_duration <= max_pause_allowed:
            pause_score = 1 - ((avg_pause_duration - 1.0) / (max_pause_allowed - 1.0))
        else:
            pause_score = 0.3  # Minimum score for excessive pauses
        scores.append(pause_score)
        
        # 4. Speech rate score (if transcript available)
        if self.transcript:
            speech_rate = self.calculate_speech_rate()
            ideal_min, ideal_max = AUDIO_CONFIG["ideal_speech_rate"]
            
            if ideal_min <= speech_rate <= ideal_max:
                rate_score = 1.0
            elif speech_rate < ideal_min:
                rate_score = max(0.3, speech_rate / ideal_min)
            else:
                rate_score = max(0.3, ideal_max / speech_rate)
            scores.append(rate_score)
        
        # Calculate weighted average
        confidence = np.mean(scores)
        
        return round(float(confidence), 3)
    
    def analyze(self) -> SpeechMetrics:
        """
        Perform complete speech analysis
        
        Returns:
            SpeechMetrics object with all analysis results
        """
        logger.info("=" * 60)
        logger.info("Starting audio analysis")
        logger.info("=" * 60)
        
        # Extract acoustic features
        acoustic_features = self.extract_acoustic_features()
        
        # Detect pauses
        pauses = self.detect_pauses()
        pause_durations = [end - start for start, end in pauses]
        
        # Analyze filler words
        filler_analysis = self.analyze_filler_words()
        
        # Calculate speech rate
        speech_rate = self.calculate_speech_rate()
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence_score(acoustic_features, pauses)
        
        # Calculate speaking duration (total - pauses)
        total_pause_duration = sum(pause_durations)
        speaking_duration = self.duration - total_pause_duration
        
        metrics = SpeechMetrics(
            speech_rate=speech_rate,
            pause_count=len(pauses),
            avg_pause_duration=round(np.mean(pause_durations), 2) if pauses else 0.0,
            max_pause_duration=round(max(pause_durations), 2) if pauses else 0.0,
            filler_words_count=filler_analysis["filler_words_count"],
            filler_ratio=filler_analysis["filler_ratio"],
            total_words=filler_analysis["total_words"],
            speaking_duration=round(speaking_duration, 2),
            confidence_score=confidence_score,
            pitch_mean=round(acoustic_features["pitch_mean"], 2),
            pitch_std=round(acoustic_features["pitch_std"], 2),
            energy_mean=round(acoustic_features["energy_mean"], 4),
            energy_std=round(acoustic_features["energy_std"], 4)
        )
        
        logger.info("=" * 60)
        logger.info("✓ Audio analysis completed")
        logger.info(f"Speech Rate: {metrics.speech_rate} WPM")
        logger.info(f"Confidence Score: {metrics.confidence_score}")
        logger.info(f"Filler Ratio: {metrics.filler_ratio:.2%}")
        logger.info("=" * 60)
        
        return metrics


def analyze_interview_audio(audio_path: str, transcript: str = "") -> Dict[str, Any]:
    """
    Convenience function to analyze interview audio
    
    Args:
        audio_path: Path to audio file
        transcript: Transcribed text
        
    Returns:
        Analysis results as dictionary
    """
    analyzer = AudioAnalyzer(audio_path, transcript)
    metrics = analyzer.analyze()
    return asdict(metrics)


if __name__ == "__main__":
    print("Audio Analysis Module - Ready")
    print(f"Ideal speech rate: {AUDIO_CONFIG['ideal_speech_rate']} WPM")

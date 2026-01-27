"""
Unit tests for Audio Analysis module
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf

from src.audio_analysis import AudioAnalyzer, analyze_interview_audio


@pytest.fixture
def sample_audio():
    """Create a temporary audio file for testing"""
    # Generate a simple sine wave (440 Hz - A note)
    duration = 3  # seconds
    sample_rate = 16000
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio_data, sample_rate)
    
    yield temp_file.name
    
    # Cleanup
    Path(temp_file.name).unlink()


def test_audio_analyzer_initialization(sample_audio):
    """Test AudioAnalyzer initialization"""
    analyzer = AudioAnalyzer(sample_audio, transcript="Hello world")
    
    assert analyzer.audio_path.exists()
    assert analyzer.transcript == "Hello world"
    assert analyzer.sr == 16000
    assert analyzer.duration > 0


def test_extract_acoustic_features(sample_audio):
    """Test acoustic feature extraction"""
    analyzer = AudioAnalyzer(sample_audio)
    features = analyzer.extract_acoustic_features()
    
    # Check all expected keys exist
    expected_keys = [
        'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
        'energy_mean', 'energy_std', 'mfcc_mean', 'mfcc_std',
        'zcr_mean', 'zcr_std'
    ]
    
    for key in expected_keys:
        assert key in features
        assert isinstance(features[key], float)


def test_detect_pauses(sample_audio):
    """Test pause detection"""
    analyzer = AudioAnalyzer(sample_audio)
    pauses = analyzer.detect_pauses()
    
    assert isinstance(pauses, list)
    # Each pause should be a tuple of (start, end)
    for pause in pauses:
        assert len(pause) == 2
        assert pause[1] > pause[0]


def test_filler_word_analysis():
    """Test filler word detection"""
    audio_path = None  # Will create dummy file
    transcript = "Um, I think, you know, like, this is basically a good approach, uh, right?"
    
    # Create a dummy audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        sr = 16000
        duration = 2
        audio_data = np.zeros(sr * duration)
        sf.write(temp_file.name, audio_data, sr)
        audio_path = temp_file.name
    
    try:
        analyzer = AudioAnalyzer(audio_path, transcript)
        filler_analysis = analyzer.analyze_filler_words()
        
        assert 'filler_words_count' in filler_analysis
        assert 'filler_ratio' in filler_analysis
        assert 'filler_breakdown' in filler_analysis
        
        # Should detect some filler words
        assert filler_analysis['filler_words_count'] > 0
        assert filler_analysis['filler_ratio'] > 0
        
    finally:
        Path(audio_path).unlink()


def test_speech_rate_calculation():
    """Test speech rate calculation"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        sr = 16000
        duration = 60  # 1 minute
        audio_data = np.zeros(sr * duration)
        sf.write(temp_file.name, audio_data, sr)
        audio_path = temp_file.name
    
    try:
        # 150 words in 1 minute = 150 WPM
        transcript = " ".join(["word"] * 150)
        analyzer = AudioAnalyzer(audio_path, transcript)
        speech_rate = analyzer.calculate_speech_rate()
        
        assert 140 <= speech_rate <= 160  # Should be around 150 WPM
        
    finally:
        Path(audio_path).unlink()


def test_confidence_score_range(sample_audio):
    """Test that confidence score is in valid range"""
    analyzer = AudioAnalyzer(sample_audio, transcript="This is a test transcript")
    
    acoustic_features = analyzer.extract_acoustic_features()
    pauses = analyzer.detect_pauses()
    
    confidence = analyzer.calculate_confidence_score(acoustic_features, pauses)
    
    assert 0 <= confidence <= 1


def test_complete_analysis(sample_audio):
    """Test complete analysis pipeline"""
    transcript = "This is a sample interview answer with good content and structure."
    
    analyzer = AudioAnalyzer(sample_audio, transcript)
    metrics = analyzer.analyze()
    
    # Check all metrics are present
    assert hasattr(metrics, 'speech_rate')
    assert hasattr(metrics, 'confidence_score')
    assert hasattr(metrics, 'filler_ratio')
    assert hasattr(metrics, 'pause_count')
    
    # Check metrics are in valid ranges
    assert 0 <= metrics.confidence_score <= 1
    assert 0 <= metrics.filler_ratio <= 1
    assert metrics.pause_count >= 0
    assert metrics.total_words > 0


def test_analyze_interview_audio_function(sample_audio):
    """Test convenience function"""
    transcript = "Test transcript"
    
    result = analyze_interview_audio(sample_audio, transcript)
    
    assert isinstance(result, dict)
    assert 'speech_rate' in result
    assert 'confidence_score' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

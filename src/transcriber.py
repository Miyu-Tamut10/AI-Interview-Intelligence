"""
Speech-to-Text Module using OpenAI Whisper
Transcribes interview audio to text for NLP analysis
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .config import MODELS_CONFIG, CACHE_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import whisper (graceful fallback)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")


class SpeechTranscriber:
    """
    Transcribes audio to text using OpenAI Whisper
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize speech transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper is not installed. Install with: pip install openai-whisper")
        
        self.model_name = model_name or MODELS_CONFIG["whisper"]["model_name"]
        
        logger.info(f"Loading Whisper model: {self.model_name}")
        
        # Load model
        self.model = whisper.load_model(
            self.model_name,
            download_root=str(CACHE_DIR)
        )
        
        logger.info(f"✓ Whisper model loaded: {self.model_name}")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: from config)
            
        Returns:
            Dictionary containing transcript and metadata
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        language = language or MODELS_CONFIG["whisper"]["language"]
        
        logger.info(f"Transcribing: {audio_path.name}")
        logger.info(f"Language: {language}")
        
        # Transcribe
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            task=MODELS_CONFIG["whisper"]["task"],
            verbose=False
        )
        
        transcript = result["text"].strip()
        
        logger.info(f"✓ Transcription completed")
        logger.info(f"Transcript length: {len(transcript)} characters")
        
        return {
            "transcript": transcript,
            "language": result.get("language", language),
            "segments": result.get("segments", []),
            "word_count": len(transcript.split())
        }


def transcribe_audio(
    audio_path: str,
    model_name: Optional[str] = None,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to transcribe audio
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
        language: Language code
        
    Returns:
        Transcription results
    """
    if not WHISPER_AVAILABLE:
        logger.error("Whisper not available - returning empty transcript")
        return {
            "transcript": "",
            "language": "en",
            "segments": [],
            "word_count": 0,
            "error": "Whisper not installed"
        }
    
    transcriber = SpeechTranscriber(model_name)
    return transcriber.transcribe(audio_path, language)


if __name__ == "__main__":
    print("Speech Transcriber Module - Ready")
    print(f"Whisper available: {WHISPER_AVAILABLE}")

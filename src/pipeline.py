"""
Main Pipeline Orchestrator
Coordinates all modules for end-to-end interview analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .config import OUTPUT_DIR
from .video_processor import VideoProcessor
from .transcriber import transcribe_audio, WHISPER_AVAILABLE
from .audio_analysis import AudioAnalyzer
from .nlp_evaluator import NLPEvaluator, TRANSFORMERS_AVAILABLE
from .face_analysis import FaceAnalyzer, MEDIAPIPE_AVAILABLE
from .scoring_engine import ScoringEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InterviewPipeline:
    """
    End-to-end pipeline for interview intelligence analysis
    """
    
    def __init__(
        self,
        video_path: str,
        question: str = "Tell me about your experience and qualifications",
        expected_keywords: Optional[list] = None
    ):
        """
        Initialize interview analysis pipeline
        
        Args:
            video_path: Path to interview video file
            question: Interview question asked
            expected_keywords: Optional list of expected keywords in answer
        """
        self.video_path = Path(video_path)
        self.question = question
        self.expected_keywords = expected_keywords or []
        
        self.results = {
            "video_path": str(self.video_path),
            "question": question,
            "expected_keywords": self.expected_keywords,
            "processing_started": datetime.now().isoformat(),
            "modules_status": {}
        }
        
        logger.info("=" * 80)
        logger.info("INTERVIEW INTELLIGENCE PIPELINE INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Video: {self.video_path.name}")
        logger.info(f"Question: {question}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete interview analysis pipeline
        
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Step 1: Process video
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1/5: VIDEO PROCESSING")
            logger.info("=" * 80)
            
            video_results = self._process_video()
            self.results["video_processing"] = video_results
            self.results["modules_status"]["video_processing"] = "success"
            
            # Step 2: Transcribe audio
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2/5: SPEECH-TO-TEXT TRANSCRIPTION")
            logger.info("=" * 80)
            
            transcription_results = self._transcribe_audio(video_results["audio_path"])
            self.results["transcription"] = transcription_results
            self.results["modules_status"]["transcription"] = "success" if WHISPER_AVAILABLE else "fallback"
            
            # Step 3: Analyze audio/speech
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3/5: AUDIO & SPEECH ANALYSIS")
            logger.info("=" * 80)
            
            speech_results = self._analyze_speech(
                video_results["audio_path"],
                transcription_results["transcript"]
            )
            self.results["speech_analysis"] = speech_results
            self.results["modules_status"]["speech_analysis"] = "success"
            
            # Step 4: Analyze answer with NLP
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4/5: NLP ANSWER EVALUATION")
            logger.info("=" * 80)
            
            nlp_results = self._evaluate_answer(transcription_results["transcript"])
            self.results["nlp_evaluation"] = nlp_results
            self.results["modules_status"]["nlp_evaluation"] = "success" if TRANSFORMERS_AVAILABLE else "fallback"
            
            # Step 5: Analyze facial engagement
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5/5: FACIAL ENGAGEMENT ANALYSIS")
            logger.info("=" * 80)
            
            facial_results = self._analyze_facial(video_results["frames_directory"])
            self.results["facial_analysis"] = facial_results
            self.results["modules_status"]["facial_analysis"] = "success" if MEDIAPIPE_AVAILABLE else "fallback"
            
            # Step 6: Generate final score
            logger.info("\n" + "=" * 80)
            logger.info("FINAL STEP: COMPUTING INTERVIEW INTELLIGENCE SCORE")
            logger.info("=" * 80)
            
            final_score = self._compute_score(nlp_results, speech_results, facial_results)
            self.results["final_score"] = final_score
            
            # Add completion timestamp
            self.results["processing_completed"] = datetime.now().isoformat()
            
            # Save results
            self._save_results()
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Final Score: {final_score['final_score']:.3f} ({final_score['grade']})")
            logger.info(f"Recommendation: {final_score['recommendation']}")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            self.results["error"] = str(e)
            self.results["status"] = "failed"
            raise
    
    def _process_video(self) -> Dict[str, Any]:
        """Process video to extract audio and frames"""
        with VideoProcessor(str(self.video_path)) as processor:
            return processor.process_video()
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio to text"""
        return transcribe_audio(audio_path)
    
    def _analyze_speech(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """Analyze speech patterns and fluency"""
        analyzer = AudioAnalyzer(audio_path, transcript)
        metrics = analyzer.analyze()
        
        # Convert dataclass to dict
        from dataclasses import asdict
        return asdict(metrics)
    
    def _evaluate_answer(self, transcript: str) -> Dict[str, Any]:
        """Evaluate answer quality with NLP"""
        evaluator = NLPEvaluator(use_embeddings=TRANSFORMERS_AVAILABLE)
        metrics = evaluator.evaluate_answer(
            transcript,
            self.question,
            self.expected_keywords if self.expected_keywords else None
        )
        
        # Convert dataclass to dict
        from dataclasses import asdict
        return asdict(metrics)
    
    def _analyze_facial(self, frames_dir: str) -> Dict[str, Any]:
        """Analyze facial engagement from video frames"""
        with FaceAnalyzer(frames_dir) as analyzer:
            metrics = analyzer.analyze_frames(max_frames=50)  # Limit for performance
            
            # Convert dataclass to dict
            from dataclasses import asdict
            return asdict(metrics)
    
    def _compute_score(
        self,
        nlp_results: Dict,
        speech_results: Dict,
        facial_results: Dict
    ) -> Dict[str, Any]:
        """Compute final interview intelligence score"""
        engine = ScoringEngine()
        score = engine.compute_final_score(nlp_results, speech_results, facial_results)
        
        # Convert dataclass to dict
        from dataclasses import asdict
        return asdict(score)
    
    def _save_results(self):
        """Save analysis results to JSON file"""
        output_file = OUTPUT_DIR / f"interview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✓ Results saved to: {output_file}")
        self.results["results_file"] = str(output_file)


def analyze_interview(
    video_path: str,
    question: str = "Tell me about your experience and qualifications",
    expected_keywords: Optional[list] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete interview analysis
    
    Args:
        video_path: Path to interview video
        question: Interview question
        expected_keywords: Expected keywords in answer
        
    Returns:
        Complete analysis results
    """
    pipeline = InterviewPipeline(video_path, question, expected_keywords)
    return pipeline.run()


if __name__ == "__main__":
    print("Interview Intelligence Pipeline - Ready")
    print(f"Whisper Available: {WHISPER_AVAILABLE}")
    print(f"Transformers Available: {TRANSFORMERS_AVAILABLE}")
    print(f"MediaPipe Available: {MEDIAPIPE_AVAILABLE}")

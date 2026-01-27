"""
Interview Intelligence Scoring Engine
Combines all module outputs to generate final interview score and insights
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .config import SCORING_WEIGHTS, THRESHOLDS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InterviewScore:
    """Data class for final interview score"""
    final_score: float
    grade: str
    nlp_score: float
    speech_score: float
    facial_score: float
    structure_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
    detailed_feedback: str
    recommendation: str
    timestamp: str


class ScoringEngine:
    """
    Hybrid rule-based + ML scoring system for interview evaluation
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize scoring engine
        
        Args:
            weights: Custom scoring weights (default: from config)
        """
        self.weights = weights or SCORING_WEIGHTS
        
        # Validate weights
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        logger.info(f"Scoring engine initialized with weights: {self.weights}")
    
    def normalize_speech_metrics(self, speech_metrics: Dict[str, Any]) -> float:
        """
        Convert speech metrics to normalized score (0-1)
        
        Args:
            speech_metrics: Dictionary from audio_analysis module
            
        Returns:
            Normalized speech score
        """
        scores = []
        
        # 1. Confidence score (already 0-1)
        confidence = speech_metrics.get("confidence_score", 0)
        scores.append(confidence)
        
        # 2. Filler ratio (inverse - lower is better)
        filler_ratio = speech_metrics.get("filler_ratio", 0)
        filler_score = max(0, 1 - (filler_ratio * 5))  # 20% filler = 0 score
        scores.append(filler_score)
        
        # 3. Speech rate score
        speech_rate = speech_metrics.get("speech_rate", 0)
        if 120 <= speech_rate <= 160:
            rate_score = 1.0
        elif 100 <= speech_rate <= 180:
            rate_score = 0.8
        else:
            rate_score = 0.6
        scores.append(rate_score)
        
        # 4. Pause appropriateness
        avg_pause = speech_metrics.get("avg_pause_duration", 0)
        if avg_pause <= 1.5:
            pause_score = 1.0
        elif avg_pause <= 3.0:
            pause_score = 0.7
        else:
            pause_score = 0.4
        scores.append(pause_score)
        
        # Weighted average
        speech_score = np.mean(scores)
        
        return round(float(speech_score), 3)
    
    def normalize_nlp_metrics(self, nlp_metrics: Dict[str, Any]) -> float:
        """
        Convert NLP metrics to normalized score (0-1)
        
        Args:
            nlp_metrics: Dictionary from nlp_evaluator module
            
        Returns:
            Normalized NLP score
        """
        # The overall_score from NLP module is already well-calibrated
        overall_score = nlp_metrics.get("overall_score", 0)
        
        # Can add additional normalization if needed
        return round(float(overall_score), 3)
    
    def normalize_facial_metrics(self, facial_metrics: Dict[str, Any]) -> float:
        """
        Convert facial metrics to normalized score (0-1)
        
        Args:
            facial_metrics: Dictionary from face_analysis module
            
        Returns:
            Normalized facial score
        """
        # The engagement_score is already well-calibrated
        engagement_score = facial_metrics.get("engagement_score", 0)
        
        # Adjust based on face detection rate
        detection_rate = facial_metrics.get("face_detection_rate", 0)
        
        # Penalize if face wasn't detected in most frames
        if detection_rate < 0.7:
            engagement_score *= 0.8
        
        return round(float(engagement_score), 3)
    
    def calculate_structure_score(
        self,
        nlp_metrics: Dict[str, Any],
        speech_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate structure score based on answer organization
        
        Args:
            nlp_metrics: NLP evaluation results
            speech_metrics: Speech analysis results
            
        Returns:
            Structure score (0-1)
        """
        scores = []
        
        # 1. Answer structure from NLP
        structure_from_nlp = nlp_metrics.get("structure_score", 0)
        scores.append(structure_from_nlp)
        
        # 2. Sentence organization
        sentence_count = nlp_metrics.get("sentence_count", 0)
        if sentence_count >= 3:
            sentence_score = 1.0
        elif sentence_count >= 2:
            sentence_score = 0.7
        else:
            sentence_score = 0.4
        scores.append(sentence_score)
        
        # 3. Speaking consistency (from pauses)
        pause_count = speech_metrics.get("pause_count", 0)
        speaking_duration = speech_metrics.get("speaking_duration", 1)
        
        # Normalize pause frequency
        pause_frequency = pause_count / (speaking_duration / 60)  # Per minute
        
        if pause_frequency <= 5:
            pause_org_score = 1.0
        elif pause_frequency <= 10:
            pause_org_score = 0.7
        else:
            pause_org_score = 0.5
        scores.append(pause_org_score)
        
        structure_score = np.mean(scores)
        return round(float(structure_score), 3)
    
    def identify_strengths(
        self,
        nlp_score: float,
        speech_score: float,
        facial_score: float,
        structure_score: float,
        nlp_metrics: Dict,
        speech_metrics: Dict,
        facial_metrics: Dict
    ) -> List[str]:
        """
        Identify candidate's strengths based on scores
        
        Returns:
            List of strength descriptions
        """
        strengths = []
        
        # NLP strengths
        if nlp_score >= 0.8:
            relevance = nlp_metrics.get("relevance_score", 0)
            clarity = nlp_metrics.get("clarity_score", 0)
            
            if relevance >= 0.8:
                strengths.append("Strong content relevance and understanding")
            if clarity >= 0.8:
                strengths.append("Excellent clarity and articulation")
        
        # Speech strengths
        if speech_score >= 0.8:
            confidence = speech_metrics.get("confidence_score", 0)
            filler_ratio = speech_metrics.get("filler_ratio", 0)
            
            if confidence >= 0.8:
                strengths.append("High vocal confidence and fluency")
            if filler_ratio < 0.05:
                strengths.append("Minimal use of filler words")
        
        # Facial strengths
        if facial_score >= 0.75:
            eye_contact = facial_metrics.get("eye_contact_ratio", 0)
            
            if eye_contact >= 0.7:
                strengths.append("Excellent eye contact and engagement")
        
        # Structure strengths
        if structure_score >= 0.8:
            strengths.append("Well-organized and structured responses")
        
        if not strengths:
            strengths.append("Demonstrates potential with room for growth")
        
        return strengths
    
    def identify_improvements(
        self,
        nlp_score: float,
        speech_score: float,
        facial_score: float,
        structure_score: float,
        nlp_metrics: Dict,
        speech_metrics: Dict,
        facial_metrics: Dict
    ) -> List[str]:
        """
        Identify areas for improvement based on scores
        
        Returns:
            List of improvement recommendations
        """
        improvements = []
        
        # NLP improvements
        if nlp_score < 0.7:
            relevance = nlp_metrics.get("relevance_score", 0)
            clarity = nlp_metrics.get("clarity_score", 0)
            
            if relevance < 0.7:
                improvements.append("Focus more directly on the question asked")
            if clarity < 0.7:
                improvements.append("Improve clarity by organizing thoughts before speaking")
        
        # Speech improvements
        if speech_score < 0.7:
            filler_ratio = speech_metrics.get("filler_ratio", 0)
            speech_rate = speech_metrics.get("speech_rate", 0)
            
            if filler_ratio > 0.1:
                improvements.append("Reduce filler words (um, uh, like)")
            if speech_rate < 100:
                improvements.append("Increase speaking pace for better engagement")
            elif speech_rate > 180:
                improvements.append("Slow down speaking pace for better clarity")
        
        # Facial improvements
        if facial_score < 0.65:
            eye_contact = facial_metrics.get("eye_contact_ratio", 0)
            
            if eye_contact < 0.6:
                improvements.append("Maintain better eye contact with the camera")
        
        # Structure improvements
        if structure_score < 0.7:
            improvements.append("Use structured approach: introduction, main points, conclusion")
        
        if not improvements:
            improvements.append("Continue refining communication skills")
        
        return improvements
    
    def generate_detailed_feedback(
        self,
        scores: Dict[str, float],
        nlp_metrics: Dict,
        speech_metrics: Dict,
        facial_metrics: Dict
    ) -> str:
        """
        Generate comprehensive feedback text
        
        Returns:
            Detailed feedback string
        """
        feedback_parts = []
        
        feedback_parts.append(f"Overall Performance: {scores['final_score']:.2%} ({scores['grade']})\n")
        
        feedback_parts.append("\n📊 Component Breakdown:")
        feedback_parts.append(f"• Content Quality (NLP): {scores['nlp_score']:.2%}")
        feedback_parts.append(f"• Speech Delivery: {scores['speech_score']:.2%}")
        feedback_parts.append(f"• Visual Engagement: {scores['facial_score']:.2%}")
        feedback_parts.append(f"• Answer Structure: {scores['structure_score']:.2%}")
        
        feedback_parts.append("\n💬 Communication Analysis:")
        feedback_parts.append(f"• Speech Rate: {speech_metrics.get('speech_rate', 0):.0f} WPM")
        feedback_parts.append(f"• Filler Word Ratio: {speech_metrics.get('filler_ratio', 0):.2%}")
        feedback_parts.append(f"• Eye Contact Quality: {facial_metrics.get('eye_contact_ratio', 0):.2%}")
        
        feedback_parts.append("\n📝 Content Analysis:")
        feedback_parts.append(nlp_metrics.get("feedback", "Good effort overall."))
        
        return "\n".join(feedback_parts)
    
    def generate_recommendation(self, final_score: float, grade: str) -> str:
        """
        Generate hiring recommendation
        
        Args:
            final_score: Final interview score
            grade: Grade (Excellent, Good, etc.)
            
        Returns:
            Recommendation string
        """
        if final_score >= THRESHOLDS["excellent"]:
            return "Strong hire - Candidate demonstrates excellent communication and technical skills"
        elif final_score >= THRESHOLDS["good"]:
            return "Hire - Candidate shows solid performance with good potential"
        elif final_score >= THRESHOLDS["average"]:
            return "Consider - Candidate shows potential but needs development in key areas"
        elif final_score >= THRESHOLDS["needs_improvement"]:
            return "Borderline - Significant improvements needed before making hiring decision"
        else:
            return "Not recommended - Candidate needs substantial improvement across multiple areas"
    
    def assign_grade(self, score: float) -> str:
        """
        Assign letter grade based on score
        
        Args:
            score: Score (0-1)
            
        Returns:
            Grade string
        """
        if score >= THRESHOLDS["excellent"]:
            return "Excellent (A)"
        elif score >= THRESHOLDS["good"]:
            return "Good (B)"
        elif score >= THRESHOLDS["average"]:
            return "Average (C)"
        elif score >= THRESHOLDS["needs_improvement"]:
            return "Needs Improvement (D)"
        else:
            return "Poor (F)"
    
    def compute_final_score(
        self,
        nlp_metrics: Dict[str, Any],
        speech_metrics: Dict[str, Any],
        facial_metrics: Dict[str, Any]
    ) -> InterviewScore:
        """
        Compute final interview intelligence score
        
        Args:
            nlp_metrics: NLP evaluation results
            speech_metrics: Speech analysis results
            facial_metrics: Facial analysis results
            
        Returns:
            InterviewScore object with complete evaluation
        """
        logger.info("=" * 60)
        logger.info("Computing final interview score")
        logger.info("=" * 60)
        
        # Normalize component scores
        nlp_score = self.normalize_nlp_metrics(nlp_metrics)
        speech_score = self.normalize_speech_metrics(speech_metrics)
        facial_score = self.normalize_facial_metrics(facial_metrics)
        structure_score = self.calculate_structure_score(nlp_metrics, speech_metrics)
        
        # Calculate weighted final score
        final_score = (
            self.weights["nlp_score"] * nlp_score +
            self.weights["speech_score"] * speech_score +
            self.weights["facial_score"] * facial_score +
            self.weights["structure_score"] * structure_score
        )
        
        final_score = round(float(final_score), 3)
        
        # Assign grade
        grade = self.assign_grade(final_score)
        
        # Identify strengths and improvements
        strengths = self.identify_strengths(
            nlp_score, speech_score, facial_score, structure_score,
            nlp_metrics, speech_metrics, facial_metrics
        )
        
        improvements = self.identify_improvements(
            nlp_score, speech_score, facial_score, structure_score,
            nlp_metrics, speech_metrics, facial_metrics
        )
        
        # Generate detailed feedback
        scores_dict = {
            "final_score": final_score,
            "grade": grade,
            "nlp_score": nlp_score,
            "speech_score": speech_score,
            "facial_score": facial_score,
            "structure_score": structure_score
        }
        
        detailed_feedback = self.generate_detailed_feedback(
            scores_dict, nlp_metrics, speech_metrics, facial_metrics
        )
        
        # Generate recommendation
        recommendation = self.generate_recommendation(final_score, grade)
        
        # Create final score object
        interview_score = InterviewScore(
            final_score=final_score,
            grade=grade,
            nlp_score=nlp_score,
            speech_score=speech_score,
            facial_score=facial_score,
            structure_score=structure_score,
            strengths=strengths,
            areas_for_improvement=improvements,
            detailed_feedback=detailed_feedback,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("=" * 60)
        logger.info(f"✓ Final Score: {final_score:.3f} ({grade})")
        logger.info(f"✓ Recommendation: {recommendation}")
        logger.info("=" * 60)
        
        return interview_score


def compute_interview_score(
    nlp_metrics: Dict[str, Any],
    speech_metrics: Dict[str, Any],
    facial_metrics: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Convenience function to compute interview score
    
    Args:
        nlp_metrics: NLP evaluation results
        speech_metrics: Speech analysis results
        facial_metrics: Facial analysis results
        weights: Optional custom weights
        
    Returns:
        Interview score as dictionary
    """
    engine = ScoringEngine(weights)
    score = engine.compute_final_score(nlp_metrics, speech_metrics, facial_metrics)
    return asdict(score)


if __name__ == "__main__":
    print("Scoring Engine Module - Ready")
    print(f"Scoring weights: {SCORING_WEIGHTS}")
    print(f"Thresholds: {THRESHOLDS}")

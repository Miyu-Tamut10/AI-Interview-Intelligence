"""
NLP Answer Evaluation Module
Evaluates quality, relevance, and clarity of interview answers using NLP
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re

from .config import MODELS_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers (graceful fallback)
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


@dataclass
class AnswerMetrics:
    """Data class for answer evaluation metrics"""
    relevance_score: float
    clarity_score: float
    structure_score: float
    keyword_coverage: float
    answer_length: int
    sentence_count: int
    avg_sentence_length: float
    technical_depth_score: float
    overall_score: float
    feedback: str


class NLPEvaluator:
    """
    Evaluates interview answers using NLP techniques
    """
    
    def __init__(self, use_embeddings: bool = True):
        """
        Initialize NLP evaluator
        
        Args:
            use_embeddings: Whether to use sentence embeddings (requires transformers)
        """
        self.use_embeddings = use_embeddings and TRANSFORMERS_AVAILABLE
        
        if self.use_embeddings:
            logger.info("Loading sentence transformer model...")
            model_name = MODELS_CONFIG["nlp"]["embedding_model"]
            self.model = SentenceTransformer(model_name)
            logger.info(f"✓ Model loaded: {model_name}")
        else:
            self.model = None
            logger.info("Running in non-embedding mode (keyword-based only)")
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not self.use_embeddings or self.model is None:
            # Fallback: simple word overlap
            return self._word_overlap_similarity(text1, text2)
        
        # Encode texts to embeddings
        embeddings = self.model.encode([text1, text2])
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback method: compute similarity based on word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def evaluate_relevance(
        self,
        answer: str,
        question: str,
        expected_keywords: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate answer relevance to the question
        
        Args:
            answer: Candidate's answer
            question: Interview question
            expected_keywords: Optional list of expected keywords
            
        Returns:
            Relevance score (0-1)
        """
        # 1. Semantic similarity with question
        semantic_score = self.compute_semantic_similarity(answer, question)
        
        # 2. Keyword coverage (if provided)
        if expected_keywords:
            keyword_score = self._compute_keyword_coverage(answer, expected_keywords)
            # Weighted combination
            relevance = 0.6 * semantic_score + 0.4 * keyword_score
        else:
            relevance = semantic_score
        
        return round(float(relevance), 3)
    
    def _compute_keyword_coverage(self, text: str, keywords: List[str]) -> float:
        """
        Compute what percentage of expected keywords appear in text
        
        Args:
            text: Text to analyze
            keywords: List of expected keywords
            
        Returns:
            Coverage ratio (0-1)
        """
        if not keywords:
            return 1.0
        
        text_lower = text.lower()
        
        # Check each keyword (support multi-word keywords)
        found_count = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Use word boundaries for better matching
            if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                found_count += 1
        
        coverage = found_count / len(keywords)
        return coverage
    
    def evaluate_clarity(self, answer: str) -> float:
        """
        Evaluate answer clarity based on structure and readability
        
        Factors:
        - Sentence structure
        - Average sentence length (not too short, not too long)
        - Use of connecting words
        - Coherence
        
        Args:
            answer: Candidate's answer
            
        Returns:
            Clarity score (0-1)
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        scores = []
        
        # 1. Sentence count score (at least 2-3 sentences for clarity)
        sentence_count = len(sentences)
        if sentence_count >= 3:
            sentence_count_score = 1.0
        elif sentence_count >= 2:
            sentence_count_score = 0.8
        else:
            sentence_count_score = 0.5
        scores.append(sentence_count_score)
        
        # 2. Average sentence length (ideal: 15-25 words)
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = np.mean(sentence_lengths)
        
        if 15 <= avg_length <= 25:
            length_score = 1.0
        elif 10 <= avg_length <= 30:
            length_score = 0.8
        else:
            length_score = 0.6
        scores.append(length_score)
        
        # 3. Use of connecting/transition words
        connecting_words = [
            'because', 'therefore', 'however', 'moreover', 'furthermore',
            'additionally', 'consequently', 'thus', 'hence', 'firstly',
            'secondly', 'finally', 'for example', 'such as', 'in addition'
        ]
        
        answer_lower = answer.lower()
        connecting_count = sum(1 for word in connecting_words if word in answer_lower)
        
        # Normalize by answer length
        words_count = len(answer.split())
        connecting_ratio = connecting_count / (words_count / 100)  # Per 100 words
        
        if connecting_ratio >= 2:
            connecting_score = 1.0
        elif connecting_ratio >= 1:
            connecting_score = 0.8
        else:
            connecting_score = 0.6
        scores.append(connecting_score)
        
        # 4. No excessive repetition
        unique_words = len(set(answer.lower().split()))
        total_words = len(answer.split())
        
        if total_words > 0:
            uniqueness_ratio = unique_words / total_words
            uniqueness_score = min(1.0, uniqueness_ratio * 1.5)
            scores.append(uniqueness_score)
        
        clarity = np.mean(scores)
        return round(float(clarity), 3)
    
    def evaluate_structure(self, answer: str) -> float:
        """
        Evaluate answer structure (introduction, body, conclusion)
        
        Args:
            answer: Candidate's answer
            
        Returns:
            Structure score (0-1)
        """
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.3  # Too short for good structure
        
        scores = []
        
        # 1. Has clear sections (at least 3 sentences)
        if len(sentences) >= 3:
            scores.append(1.0)
        elif len(sentences) >= 2:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # 2. Logical flow indicators
        first_sentence = sentences[0].lower()
        last_sentence = sentences[-1].lower() if sentences else ""
        
        # Check for introduction phrases
        intro_phrases = ['in my experience', 'i believe', 'i think', 'first', 'to begin']
        has_intro = any(phrase in first_sentence for phrase in intro_phrases)
        
        # Check for conclusion phrases
        conclusion_phrases = ['in conclusion', 'therefore', 'thus', 'finally', 'overall', 'in summary']
        has_conclusion = any(phrase in last_sentence for phrase in conclusion_phrases)
        
        flow_score = 0.5
        if has_intro:
            flow_score += 0.25
        if has_conclusion:
            flow_score += 0.25
        
        scores.append(flow_score)
        
        structure = np.mean(scores)
        return round(float(structure), 3)
    
    def evaluate_technical_depth(self, answer: str, domain: str = "general") -> float:
        """
        Evaluate technical depth based on vocabulary and concepts
        
        Args:
            answer: Candidate's answer
            domain: Domain of interview (general, technical, etc.)
            
        Returns:
            Technical depth score (0-1)
        """
        # Technical/professional vocabulary indicators
        technical_indicators = [
            'algorithm', 'system', 'architecture', 'implement', 'optimize',
            'framework', 'methodology', 'approach', 'strategy', 'analysis',
            'evaluate', 'design', 'develop', 'integrate', 'process',
            'performance', 'efficiency', 'scalability', 'solution', 'requirements'
        ]
        
        answer_lower = answer.lower()
        
        # Count technical terms
        technical_count = sum(1 for term in technical_indicators if term in answer_lower)
        
        # Normalize by answer length
        words_count = len(answer.split())
        if words_count == 0:
            return 0.0
        
        technical_ratio = technical_count / (words_count / 50)  # Per 50 words
        
        # Score based on technical density
        if technical_ratio >= 2:
            technical_score = 1.0
        elif technical_ratio >= 1:
            technical_score = 0.8
        elif technical_ratio >= 0.5:
            technical_score = 0.6
        else:
            technical_score = 0.4
        
        return round(float(technical_score), 3)
    
    def generate_feedback(self, metrics: Dict[str, Any]) -> str:
        """
        Generate human-readable feedback based on metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            Feedback string
        """
        feedback_parts = []
        
        # Relevance feedback
        relevance = metrics.get("relevance_score", 0)
        if relevance >= 0.8:
            feedback_parts.append("Excellent relevance to the question.")
        elif relevance >= 0.6:
            feedback_parts.append("Good understanding, but could be more focused.")
        else:
            feedback_parts.append("Answer lacks relevance; address the question more directly.")
        
        # Clarity feedback
        clarity = metrics.get("clarity_score", 0)
        if clarity >= 0.8:
            feedback_parts.append("Clear and well-articulated response.")
        elif clarity >= 0.6:
            feedback_parts.append("Generally clear, but could improve sentence structure.")
        else:
            feedback_parts.append("Improve clarity by organizing thoughts better.")
        
        # Structure feedback
        structure = metrics.get("structure_score", 0)
        if structure >= 0.8:
            feedback_parts.append("Well-structured answer with logical flow.")
        elif structure >= 0.6:
            feedback_parts.append("Decent structure, consider adding intro/conclusion.")
        else:
            feedback_parts.append("Needs better structure; use clear beginning and end.")
        
        # Technical depth feedback
        technical = metrics.get("technical_depth_score", 0)
        if technical >= 0.8:
            feedback_parts.append("Strong technical depth demonstrated.")
        elif technical >= 0.6:
            feedback_parts.append("Moderate technical detail; could elaborate more.")
        else:
            feedback_parts.append("Include more specific technical details and examples.")
        
        return " ".join(feedback_parts)
    
    def evaluate_answer(
        self,
        answer: str,
        question: str,
        expected_keywords: Optional[List[str]] = None
    ) -> AnswerMetrics:
        """
        Perform complete answer evaluation
        
        Args:
            answer: Candidate's answer
            question: Interview question
            expected_keywords: Optional expected keywords
            
        Returns:
            AnswerMetrics object with all evaluation results
        """
        logger.info("Evaluating answer quality...")
        
        # Compute all metrics
        relevance_score = self.evaluate_relevance(answer, question, expected_keywords)
        clarity_score = self.evaluate_clarity(answer)
        structure_score = self.evaluate_structure(answer)
        technical_depth_score = self.evaluate_technical_depth(answer)
        
        # Keyword coverage
        if expected_keywords:
            keyword_coverage = self._compute_keyword_coverage(answer, expected_keywords)
        else:
            keyword_coverage = 1.0
        
        # Basic text statistics
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        words = answer.split()
        answer_length = len(words)
        avg_sentence_length = answer_length / sentence_count if sentence_count > 0 else 0
        
        # Compute overall score (weighted average)
        overall_score = (
            0.35 * relevance_score +
            0.25 * clarity_score +
            0.20 * structure_score +
            0.20 * technical_depth_score
        )
        
        metrics_dict = {
            "relevance_score": relevance_score,
            "clarity_score": clarity_score,
            "structure_score": structure_score,
            "technical_depth_score": technical_depth_score,
            "keyword_coverage": round(keyword_coverage, 3),
            "answer_length": answer_length,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "overall_score": round(overall_score, 3)
        }
        
        # Generate feedback
        feedback = self.generate_feedback(metrics_dict)
        
        metrics = AnswerMetrics(
            **metrics_dict,
            feedback=feedback
        )
        
        logger.info(f"✓ Answer evaluated - Overall: {overall_score:.3f}")
        
        return metrics


def evaluate_interview_answer(
    answer: str,
    question: str,
    expected_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate interview answer
    
    Args:
        answer: Candidate's answer
        question: Interview question
        expected_keywords: Optional expected keywords
        
    Returns:
        Evaluation results as dictionary
    """
    evaluator = NLPEvaluator(use_embeddings=TRANSFORMERS_AVAILABLE)
    metrics = evaluator.evaluate_answer(answer, question, expected_keywords)
    return asdict(metrics)


if __name__ == "__main__":
    print("NLP Evaluator Module - Ready")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")

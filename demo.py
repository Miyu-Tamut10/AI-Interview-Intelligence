"""
Demo script to showcase the Interview Intelligence System
Run this to test the system without the web interface
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import validate_config


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🎯  AI-Powered Interview Intelligence System  🎯       ║
    ║                                                           ║
    ║   Professional Multimodal Interview Evaluation Platform   ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_system():
    """Check system requirements and module availability"""
    print("\n" + "=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)
    
    # Check Python version
    import sys
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check core dependencies
    dependencies = {
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "librosa": "Librosa",
        "streamlit": "Streamlit"
    }
    
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {display_name} installed")
        except ImportError:
            print(f"✗ {display_name} not installed")
    
    # Check optional dependencies
    print("\nOptional AI Models:")
    
    try:
        import whisper
        print("✓ Whisper (Speech-to-Text)")
    except ImportError:
        print("✗ Whisper not installed - install with: pip install openai-whisper")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence Transformers (NLP)")
    except ImportError:
        print("✗ Sentence Transformers not installed - install with: pip install sentence-transformers")
    
    try:
        import mediapipe
        print("✓ MediaPipe (Face Analysis)")
    except ImportError:
        print("✗ MediaPipe not installed - install with: pip install mediapipe")
    
    # Validate configuration
    print("\nConfiguration:")
    try:
        validate_config()
        print("✓ Configuration valid")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
    
    print("=" * 60)


def show_architecture():
    """Display system architecture"""
    print("\n" + "=" * 60)
    print("SYSTEM ARCHITECTURE")
    print("=" * 60)
    
    arch = """
    Video Input
        ↓
    [Video Processor] → Extract Audio + Frames
        ↓                      ↓
    [Transcriber]         [Face Analyzer]
        ↓                      ↓
    [Audio Analyzer]      [Engagement Metrics]
        ↓                      ↓
    [NLP Evaluator] ← Transcript
        ↓
    [Scoring Engine] ← Combines All Signals
        ↓
    Final Report + Recommendations
    """
    print(arch)
    print("=" * 60)


def show_module_info():
    """Show information about each module"""
    print("\n" + "=" * 60)
    print("MODULE OVERVIEW")
    print("=" * 60)
    
    modules = {
        "video_processor.py": "Extracts audio and frames from video files",
        "transcriber.py": "Converts speech to text using Whisper",
        "audio_analysis.py": "Analyzes speech patterns, fluency, confidence",
        "nlp_evaluator.py": "Evaluates answer quality, relevance, clarity",
        "face_analysis.py": "Analyzes facial engagement and eye contact",
        "scoring_engine.py": "Combines all signals into final score"
    }
    
    for module, description in modules.items():
        print(f"\n📄 {module}")
        print(f"   {description}")
    
    print("\n" + "=" * 60)


def show_usage():
    """Show usage examples"""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n1. Web Application (Recommended):")
    print("   $ streamlit run app.py")
    
    print("\n2. Python API:")
    print("""
    from src.pipeline import analyze_interview
    
    results = analyze_interview(
        video_path="interview.mp4",
        question="Tell me about your experience",
        expected_keywords=["python", "teamwork"]
    )
    
    print(f"Score: {results['final_score']['final_score']:.2%}")
    """)
    
    print("\n3. Command Line:")
    print("""
    python -c "from src.pipeline import analyze_interview; \\
               analyze_interview('video.mp4')"
    """)
    
    print("\n" + "=" * 60)


def show_scoring():
    """Show scoring methodology"""
    print("\n" + "=" * 60)
    print("SCORING METHODOLOGY")
    print("=" * 60)
    
    print("""
    Final Score = Weighted Average of:
    
    • NLP Score (35%)
      - Content relevance to question
      - Answer clarity and structure
      - Technical depth
      - Keyword coverage
    
    • Speech Score (30%)
      - Vocal confidence
      - Speech rate (optimal: 120-160 WPM)
      - Filler word usage
      - Pause patterns
    
    • Facial Score (20%)
      - Eye contact quality
      - Head stability
      - Visual engagement
      - Face detection consistency
    
    • Structure Score (15%)
      - Answer organization
      - Logical flow
      - Introduction and conclusion
      - Coherence
    
    Grading Scale:
    A (Excellent): 85%+
    B (Good): 70-84%
    C (Average): 55-69%
    D (Needs Improvement): 40-54%
    F (Poor): <40%
    """)
    
    print("=" * 60)


def main():
    """Main demo function"""
    print_banner()
    check_system()
    show_architecture()
    show_module_info()
    show_scoring()
    show_usage()
    
    print("\n" + "=" * 60)
    print("READY TO START!")
    print("=" * 60)
    print("\nTo begin analysis:")
    print("  1. Run: streamlit run app.py")
    print("  2. Upload an interview video")
    print("  3. View comprehensive results\n")
    print("For questions or issues:")
    print("  - Check README.md for documentation")
    print("  - Review src/config.py for settings")
    print("  - Run tests with: pytest tests/")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

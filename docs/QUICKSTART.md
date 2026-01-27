# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Interview-Intelligence.git
cd AI-Interview-Intelligence

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation (30 seconds)

```bash
# Run the demo script
python demo.py
```

You should see a system check showing which components are installed.

### Step 3: Launch Web Application (30 seconds)

```bash
# Start Streamlit app
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

### Step 4: Analyze Your First Interview (2 minutes)

1. **Upload Video**: Click "Browse files" and select an interview video
2. **Enter Question**: Type the interview question that was asked
3. **Add Keywords** (optional): Enter expected keywords (comma-separated)
4. **Click Analyze**: Wait for processing (~1-2 minutes for a 5-minute video)
5. **View Results**: See comprehensive scores, feedback, and recommendations

## 📹 Sample Test Video

Don't have an interview video? Record a quick test video:

### Using Your Webcam

```python
# Quick test recording script
import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Record video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_interview.mp4', fourcc, 30.0, (1280, 720))

print("Recording... Press 'q' to stop")

while True:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Recording', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as test_interview.mp4")
```

### Sample Interview Questions

Use these for testing:

**Technical:**
- "Tell me about your experience with Python and machine learning."
- "Describe a challenging project you worked on and how you solved it."

**Behavioral:**
- "Tell me about a time you demonstrated leadership."
- "How do you handle conflict in a team setting?"

**General:**
- "Tell me about yourself and your qualifications for this role."
- "Where do you see yourself in five years?"

## 🎯 Example API Usage

### Basic Analysis

```python
from src.pipeline import analyze_interview

# Run analysis
results = analyze_interview(
    video_path="interview_video.mp4",
    question="Tell me about your experience with Python",
    expected_keywords=["python", "programming", "projects"]
)

# Print results
print(f"Final Score: {results['final_score']['final_score']:.2%}")
print(f"Grade: {results['final_score']['grade']}")
print(f"\nStrengths:")
for strength in results['final_score']['strengths']:
    print(f"  - {strength}")
```

### Module-by-Module

```python
from src.video_processor import VideoProcessor
from src.transcriber import transcribe_audio
from src.audio_analysis import AudioAnalyzer
from src.nlp_evaluator import NLPEvaluator

# 1. Process video
with VideoProcessor("interview.mp4") as processor:
    video_data = processor.process_video()
    print(f"✓ Extracted audio: {video_data['audio_path']}")

# 2. Transcribe
transcription = transcribe_audio(video_data["audio_path"])
print(f"✓ Transcript: {transcription['transcript'][:100]}...")

# 3. Analyze speech
analyzer = AudioAnalyzer(video_data["audio_path"], transcription["transcript"])
speech_metrics = analyzer.analyze()
print(f"✓ Speech Rate: {speech_metrics.speech_rate} WPM")

# 4. Evaluate content
evaluator = NLPEvaluator()
nlp_metrics = evaluator.evaluate_answer(
    transcription["transcript"],
    "Your interview question"
)
print(f"✓ Relevance Score: {nlp_metrics.relevance_score:.2%}")
```

## 🔧 Configuration

### Adjust Model Settings

Edit `src/config.py`:

```python
# Use larger Whisper model for better accuracy
MODELS_CONFIG = {
    "whisper": {
        "model_name": "small",  # Change from "base" to "small" or "medium"
    }
}

# Adjust scoring weights
SCORING_WEIGHTS = {
    "nlp_score": 0.40,      # Increase emphasis on content
    "speech_score": 0.30,
    "facial_score": 0.15,   # Decrease emphasis on visual
    "structure_score": 0.15
}
```

### Common Issues & Solutions

#### Issue: "FFmpeg not found"
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

#### Issue: "CUDA out of memory"
```python
# Use smaller Whisper model
MODELS_CONFIG["whisper"]["model_name"] = "tiny"  # or "base"
```

#### Issue: "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📊 Understanding Results

### Score Interpretation

| Score | Grade | Meaning | Action |
|-------|-------|---------|--------|
| 85-100% | A | Excellent | Strong hire |
| 70-84% | B | Good | Recommend hire |
| 55-69% | C | Average | Consider carefully |
| 40-54% | D | Below average | Needs improvement |
| 0-39% | F | Poor | Not recommended |

### Component Breakdown

**NLP Score (35% weight)**
- Measures answer relevance, clarity, and depth
- High score = answers questions directly with clear explanations

**Speech Score (30% weight)**
- Evaluates vocal delivery and fluency
- High score = confident tone, good pacing, minimal fillers

**Facial Score (20% weight)**
- Assesses visual engagement
- High score = good eye contact, stable positioning

**Structure Score (15% weight)**
- Analyzes answer organization
- High score = logical flow, clear beginning and end

## 📝 Best Practices

### For Best Results

**Video Quality:**
- ✅ Good lighting (face clearly visible)
- ✅ Stable camera position
- ✅ Minimal background noise
- ✅ Face centered in frame
- ✅ Duration: 2-10 minutes

**Interview Setup:**
- ✅ Clear question stated
- ✅ Single speaker (candidate)
- ✅ Answer-focused (not conversation)
- ✅ Professional setting

### What to Avoid

- ❌ Very short videos (<30 seconds)
- ❌ Multiple speakers talking over each other
- ❌ Poor audio quality (background noise, echo)
- ❌ Face not visible or off-camera
- ❌ Extremely low resolution

## 🎓 Next Steps

1. **Explore the Code**: Check out individual modules in `src/`
2. **Read Documentation**: See `docs/ARCHITECTURE.md` for details
3. **Run Tests**: Execute `pytest tests/` to verify functionality
4. **Customize**: Modify `src/config.py` for your needs
5. **Integrate**: Use the API in your hiring workflow

## 💡 Tips for Interviewers

### Using Results Effectively

1. **Don't Rely Solely on Scores**: Use as one data point among many
2. **Read the Feedback**: Qualitative insights are valuable
3. **Compare Candidates**: Consistent evaluation across all candidates
4. **Consider Context**: Technical interviews vs behavioral interviews
5. **Follow Up**: Use identified weaknesses for targeted follow-up questions

### Customization Ideas

- Adjust weights based on role requirements
- Add domain-specific keywords for technical roles
- Create custom rubrics for different positions
- Integrate with your ATS (Applicant Tracking System)

## 🆘 Getting Help

- **Documentation**: Check `README.md` and `docs/`
- **Issues**: Open a GitHub issue
- **Discussions**: Join GitHub Discussions
- **Email**: support@example.com

---

**Ready to go? Run `streamlit run app.py` and start analyzing!** 🚀

# 🎯 AI-Powered Multimodal Interview Intelligence System

A **production-ready AI system** that analyzes recorded interview videos using **speech, text, and facial cues** to generate **objective, explainable interview evaluation reports**.

This project focuses on **real-world usability**, **clean architecture**, and **reproducibility**, not just model accuracy.

---

## 🚀 Why This Project Exists

Interview evaluation is often:
- subjective  
- inconsistent  
- biased  
- hard to scale  

This system provides a **structured, data-driven alternative** by analyzing:
- how a candidate **speaks**
- what the candidate **says**
- how engaged the candidate **appears**

The result is a **clear interview intelligence report** that recruiters can actually use.

---

## 🧠 What the System Does

1. Accepts a **recorded interview video**
2. Extracts **audio and video frames**
3. Analyzes:
   - 🎤 **Speech** (confidence, pace, fillers)
   - 📝 **Answer quality** (relevance, clarity)
   - 👁️ **Facial engagement** (eye contact, stability)
4. Combines all signals into a **final interview score**
5. Generates **human-readable feedback**

---

## 🏗️ High-Level Architecture

```
Interview Video
     ↓
Video Processor (Audio + Frames)
     ↓
Speech Analysis  ←→  NLP Evaluation  ←→  Facial Analysis
     ↓
Hybrid Scoring Engine
     ↓
Interview Intelligence Report
```

Each module is **independent, explainable, and testable**.

---

## 🧪 Key Features

- Multimodal AI (Audio + NLP + Vision)
- OpenAI Whisper for speech-to-text
- Transformer embeddings for semantic analysis
- MediaPipe Face Mesh for engagement analysis
- Hybrid rule + ML scoring
- Clean Streamlit UI for recruiters
- Modular, production-style codebase

---

## 🛠️ Tech Stack

### Backend / AI
- Python 3.10
- PyTorch
- Hugging Face Transformers
- OpenAI Whisper
- Librosa
- MediaPipe
- OpenCV
- NumPy / SciPy
- FFmpeg

### Frontend
- Streamlit

### Dev & Quality
- Git & GitHub
- Pytest
- Black
- Flake8

---

## 🐍 Python Version (IMPORTANT)

This project **requires Python 3.10.x**.

> Newer Python versions (3.11+) may cause incompatibilities with  
> PyTorch, MediaPipe, Librosa, and Whisper.

**Verified working version**
- Python **3.10.11**

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10.x
- FFmpeg installed and added to PATH

### Setup

```bash
git clone https://github.com/your-username/AI-Interview-Intelligence.git
cd AI-Interview-Intelligence

py -3.10 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Then open:
```
http://localhost:8501
```

---

## ▶️ How to Use

1. Launch the Streamlit app
2. Upload an interview video (MP4 / MOV / AVI)
3. Enter the interview question
4. (Optional) Add expected keywords
5. Click **Analyze Interview**
6. View scores, breakdowns, and feedback

---

## 📊 Output You Get

- Final Interview Score (A/B/C style grading)
- Speech metrics (WPM, filler ratio, confidence)
- NLP scores (relevance, clarity)
- Facial engagement score
- Strengths & improvement areas
- Hiring-style recommendation

---

## ⚠️ Limitations

- Designed for **single-speaker** interviews
- Requires **reasonable lighting & audio**
- Batch processing (not real-time yet)
- Optimized for interviews up to ~10 minutes

---

## 🔮 Future Improvements

- Real-time interview analysis
- FastAPI backend
- ATS (Applicant Tracking System) integration
- Emotion recognition (voice + face)
- Multi-speaker support
- Cloud deployment

---

## 🧑‍💻 Why This Is Different from Typical ML Projects

- Not a notebook-only demo
- Modular, production-style architecture
- Explainable scoring (not black-box)
- Actually usable by non-technical users
- Built with deployment and reproducibility in mind

---

## 📄 License

MIT License — free to use, modify, and extend.

---

## 🎤 Interview-Ready Summary

> “I built a multimodal AI system that evaluates interview performance by combining speech analysis, NLP-based answer evaluation, and facial engagement analysis, producing explainable hiring intelligence rather than just raw predictions.”

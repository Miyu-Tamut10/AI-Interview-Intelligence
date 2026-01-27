# System Architecture

## Overview

The AI-Powered Interview Intelligence System follows a modular pipeline architecture where each component performs a specialized task in the analysis workflow.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                                │
│                  Interview Video File                           │
│              (.mp4, .avi, .mov, .mkv, .webm)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                           │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │         Video Processor (video_processor.py)           │   │
│  │  • FFmpeg-based audio extraction (16kHz mono)          │   │
│  │  • Frame sampling (configurable FPS)                   │   │
│  │  • Video metadata extraction                           │   │
│  └────────────┬───────────────────────┬────────────────────┘   │
│               │                       │                         │
└───────────────┼───────────────────────┼─────────────────────────┘
                │                       │
      ┌─────────▼─────────┐   ┌─────────▼──────────┐
      │  Audio Stream     │   │   Frame Sequence   │
      │   (.wav file)     │   │  (JPG images)      │
      └─────────┬─────────┘   └─────────┬──────────┘
                │                       │
                ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYSIS LAYER                                │
│                                                                 │
│  ┌─────────────────────┐         ┌────────────────────────┐   │
│  │   Transcriber       │         │   Face Analyzer        │   │
│  │  (transcriber.py)   │         │  (face_analysis.py)    │   │
│  │                     │         │                        │   │
│  │  • Whisper STT      │         │  • MediaPipe Face Mesh │   │
│  │  • Language detect  │         │  • Eye contact ratio   │   │
│  │  • Timestamped text │         │  • Head stability      │   │
│  └──────────┬──────────┘         │  • Engagement score    │   │
│             │                     └────────────┬───────────┘   │
│             ▼                                  │               │
│  ┌─────────────────────┐                      │               │
│  │  Audio Analyzer     │                      │               │
│  │ (audio_analysis.py) │                      │               │
│  │                     │                      │               │
│  │  • Librosa features │                      │               │
│  │  • Speech rate      │                      │               │
│  │  • Confidence score │                      │               │
│  │  • Filler detection │                      │               │
│  │  • Pause analysis   │                      │               │
│  └──────────┬──────────┘                      │               │
│             │                                  │               │
│             ▼                                  │               │
│  ┌─────────────────────┐                      │               │
│  │   NLP Evaluator     │                      │               │
│  │ (nlp_evaluator.py)  │                      │               │
│  │                     │                      │               │
│  │  • BERT embeddings  │                      │               │
│  │  • Semantic sim.    │                      │               │
│  │  • Clarity score    │                      │               │
│  │  • Structure score  │                      │               │
│  │  • Keyword coverage │                      │               │
│  └──────────┬──────────┘                      │               │
│             │                                  │               │
└─────────────┼──────────────────────────────────┼───────────────┘
              │                                  │
              └──────────┬───────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCORING LAYER                                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │          Scoring Engine (scoring_engine.py)              │ │
│  │                                                          │ │
│  │  Hybrid Rule-Based + ML Scoring:                        │ │
│  │                                                          │ │
│  │  Final Score = 0.35×NLP + 0.30×Speech +                │ │
│  │                0.20×Facial + 0.15×Structure             │ │
│  │                                                          │ │
│  │  • Grade assignment (A-F)                               │ │
│  │  • Strength identification                              │ │
│  │  • Improvement recommendations                          │ │
│  │  • Hiring recommendation                                │ │
│  └──────────────────────────┬───────────────────────────────┘ │
│                             │                                  │
└─────────────────────────────┼──────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Interview Intelligence Report               │ │
│  │                                                          │ │
│  │  • Final Score: 0.781 (Good - B)                        │ │
│  │  • Component Scores: NLP, Speech, Facial, Structure     │ │
│  │  • Strengths & Improvements                             │ │
│  │  • Detailed Feedback                                    │ │
│  │  • Hiring Recommendation                                │ │
│  │  • Full Transcript                                      │ │
│  │  • Exportable JSON Report                               │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Video Processor
- **Input**: Video file (MP4, AVI, MOV, etc.)
- **Output**: Audio (WAV) + Frame sequence (JPG)
- **Technology**: FFmpeg, OpenCV
- **Processing Time**: ~5-10 seconds for 5-minute video

### 2. Speech-to-Text (Transcriber)
- **Input**: Audio file (WAV)
- **Output**: Text transcript with timestamps
- **Technology**: OpenAI Whisper
- **Model Size**: Base (configurable)
- **Processing Time**: ~30 seconds for 5-minute audio

### 3. Audio Analyzer
- **Input**: Audio file + Transcript
- **Output**: Speech metrics (rate, confidence, fillers)
- **Technology**: Librosa, NumPy
- **Features**: 13 MFCC, pitch, energy, ZCR
- **Processing Time**: ~10 seconds

### 4. NLP Evaluator
- **Input**: Transcript + Question + Keywords
- **Output**: Content quality scores
- **Technology**: Sentence-BERT, Transformers
- **Metrics**: Relevance, clarity, structure, depth
- **Processing Time**: ~5 seconds

### 5. Face Analyzer
- **Input**: Frame sequence (JPG files)
- **Output**: Engagement metrics
- **Technology**: MediaPipe Face Mesh, OpenCV
- **Metrics**: Eye contact, head stability, engagement
- **Processing Time**: ~15 seconds (50 frames)

### 6. Scoring Engine
- **Input**: All module outputs
- **Output**: Final score + Recommendations
- **Technology**: Weighted scoring algorithm
- **Logic**: Hybrid rule-based + normalization
- **Processing Time**: <1 second

## Data Flow

```
Video File
    ↓
[Extract] → Audio + Frames
    ↓
[Transcribe] → Text
    ↓
[Analyze] → {Speech, NLP, Facial} Metrics
    ↓
[Score] → Final Report
```

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Video Processing | FFmpeg, OpenCV |
| Speech Recognition | OpenAI Whisper |
| Audio Analysis | Librosa, NumPy, SciPy |
| NLP | BERT, Sentence-Transformers |
| Computer Vision | MediaPipe, OpenCV |
| ML Framework | PyTorch |
| Web Interface | Streamlit |
| Orchestration | Python, Threading |

## Scalability Considerations

### Current Architecture
- **Processing**: Sequential (one video at a time)
- **Storage**: Local filesystem
- **Deployment**: Single machine

### Future Enhancements
- **Parallel Processing**: Queue-based system (Celery + Redis)
- **Distributed Storage**: S3/GCS for video files
- **Microservices**: Each module as independent service
- **Container Orchestration**: Docker + Kubernetes
- **Load Balancing**: NGINX for web tier
- **Database**: PostgreSQL for results storage
- **Caching**: Redis for intermediate results

## Performance Metrics

| Video Duration | Processing Time | CPU Usage | Memory Usage |
|----------------|----------------|-----------|--------------|
| 2 minutes | ~45 seconds | 60-80% | ~2 GB |
| 5 minutes | ~90 seconds | 70-90% | ~3 GB |
| 10 minutes | ~3 minutes | 80-100% | ~4 GB |

*Tested on: Intel i7-9700K, 16GB RAM, no GPU*

## Error Handling

Each module includes:
- Input validation
- Graceful degradation (fallback modes)
- Detailed logging
- Exception handling with context
- User-friendly error messages

## Configuration Management

All configurations centralized in `src/config.py`:
- Model selection
- Scoring weights
- Thresholds
- File paths
- Processing parameters

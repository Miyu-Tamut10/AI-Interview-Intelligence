"""
Streamlit Web Application for Interview Intelligence System
Professional UI for HR recruiters
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import analyze_interview
from src.config import OUTPUT_DIR, VIDEO_CONFIG

# Page configuration
st.set_page_config(
    page_title="AI Interview Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .score-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .score-good {
        color: #17a2b8;
        font-weight: bold;
    }
    .score-average {
        color: #ffc107;
        font-weight: bold;
    }
    .score-poor {
        color: #dc3545;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def display_header():
    """Display application header"""
    st.markdown('<div class="main-header">🎯 AI-Powered Interview Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### Professional Multimodal Interview Evaluation Platform
    
    This system analyzes candidate interviews using advanced AI to evaluate:
    - 📝 **Content Quality** (NLP Analysis)
    - 🎤 **Speech Delivery** (Audio Analysis)
    - 👁️ **Visual Engagement** (Facial Analysis)
    - 📊 **Answer Structure** (Organization)
    """)
    st.markdown("---")


def sidebar_config():
    """Configure sidebar inputs"""
    st.sidebar.title("⚙️ Configuration")
    
    # File upload
    st.sidebar.header("1. Upload Video")
    video_file = st.sidebar.file_uploader(
        "Upload Interview Video",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Upload a recorded interview video (max 10 minutes)"
    )
    
    # Question input
    st.sidebar.header("2. Interview Details")
    question = st.sidebar.text_area(
        "Interview Question",
        value="Tell me about your experience and qualifications for this role.",
        height=100,
        help="Enter the question that was asked in the interview"
    )
    
    # Expected keywords (optional)
    st.sidebar.header("3. Expected Keywords (Optional)")
    keywords_input = st.sidebar.text_input(
        "Keywords (comma-separated)",
        placeholder="e.g., python, machine learning, teamwork",
        help="Enter keywords you expect in a good answer (optional)"
    )
    
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()] if keywords_input else None
    
    return video_file, question, keywords


def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily"""
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def get_score_class(score):
    """Get CSS class based on score"""
    if score >= 0.85:
        return "score-excellent"
    elif score >= 0.70:
        return "score-good"
    elif score >= 0.55:
        return "score-average"
    else:
        return "score-poor"


def display_results(results):
    """Display analysis results"""
    
    # Extract final score
    final_score_data = results.get("final_score", {})
    final_score = final_score_data.get("final_score", 0)
    grade = final_score_data.get("grade", "N/A")
    
    # Main score display
    st.markdown("## 📊 Interview Analysis Results")
    st.markdown("---")
    
    # Overall score card
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### Overall Performance")
        score_class = get_score_class(final_score)
        st.markdown(f'<h1 class="{score_class}">{final_score:.1%}</h1>', unsafe_allow_html=True)
        st.markdown(f"**Grade:** {grade}")
    
    with col2:
        st.metric("NLP Score", f"{final_score_data.get('nlp_score', 0):.1%}")
        st.metric("Speech Score", f"{final_score_data.get('speech_score', 0):.1%}")
    
    with col3:
        st.metric("Facial Score", f"{final_score_data.get('facial_score', 0):.1%}")
        st.metric("Structure Score", f"{final_score_data.get('structure_score', 0):.1%}")
    
    st.markdown("---")
    
    # Component breakdown with progress bars
    st.markdown("### 📈 Component Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Content Quality (NLP)**")
        st.progress(final_score_data.get('nlp_score', 0))
        
        st.markdown("**Speech Delivery**")
        st.progress(final_score_data.get('speech_score', 0))
    
    with col2:
        st.markdown("**Visual Engagement**")
        st.progress(final_score_data.get('facial_score', 0))
        
        st.markdown("**Answer Structure**")
        st.progress(final_score_data.get('structure_score', 0))
    
    st.markdown("---")
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Strengths")
        strengths = final_score_data.get("strengths", [])
        for strength in strengths:
            st.markdown(f"- {strength}")
    
    with col2:
        st.markdown("### 📈 Areas for Improvement")
        improvements = final_score_data.get("areas_for_improvement", [])
        for improvement in improvements:
            st.markdown(f"- {improvement}")
    
    st.markdown("---")
    
    # Detailed feedback
    st.markdown("### 📝 Detailed Analysis")
    
    with st.expander("View Complete Feedback", expanded=True):
        feedback = final_score_data.get("detailed_feedback", "No feedback available")
        st.text(feedback)
    
    # Speech metrics
    with st.expander("🎤 Speech Analysis Details"):
        speech_data = results.get("speech_analysis", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Speech Rate", f"{speech_data.get('speech_rate', 0):.0f} WPM")
            st.metric("Confidence", f"{speech_data.get('confidence_score', 0):.1%}")
        
        with col2:
            st.metric("Filler Words", f"{speech_data.get('filler_words_count', 0)}")
            st.metric("Filler Ratio", f"{speech_data.get('filler_ratio', 0):.2%}")
        
        with col3:
            st.metric("Pause Count", f"{speech_data.get('pause_count', 0)}")
            st.metric("Avg Pause", f"{speech_data.get('avg_pause_duration', 0):.1f}s")
    
    # NLP metrics
    with st.expander("📝 Content Quality Details"):
        nlp_data = results.get("nlp_evaluation", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Relevance", f"{nlp_data.get('relevance_score', 0):.1%}")
            st.metric("Clarity", f"{nlp_data.get('clarity_score', 0):.1%}")
        
        with col2:
            st.metric("Structure", f"{nlp_data.get('structure_score', 0):.1%}")
            st.metric("Technical Depth", f"{nlp_data.get('technical_depth_score', 0):.1%}")
        
        with col3:
            st.metric("Word Count", f"{nlp_data.get('answer_length', 0)}")
            st.metric("Sentences", f"{nlp_data.get('sentence_count', 0)}")
    
    # Facial metrics
    with st.expander("👁️ Visual Engagement Details"):
        facial_data = results.get("facial_analysis", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Eye Contact", f"{facial_data.get('eye_contact_ratio', 0):.1%}")
            st.metric("Engagement", f"{facial_data.get('engagement_score', 0):.1%}")
        
        with col2:
            st.metric("Head Stability", f"{facial_data.get('head_stability_score', 0):.1%}")
            st.metric("Frames Analyzed", f"{facial_data.get('frames_analyzed', 0)}")
        
        with col3:
            st.metric("Face Detection", f"{facial_data.get('face_detection_rate', 0):.1%}")
            st.metric("Frames w/ Face", f"{facial_data.get('frames_with_face', 0)}")
    
    # Transcript
    with st.expander("📄 Interview Transcript"):
        transcript_data = results.get("transcription", {})
        transcript = transcript_data.get("transcript", "No transcript available")
        st.text_area("Transcript", transcript, height=200)
    
    st.markdown("---")
    
    # Recommendation
    st.markdown("### 🎯 Hiring Recommendation")
    recommendation = final_score_data.get("recommendation", "No recommendation available")
    
    if final_score >= 0.85:
        st.success(f"✅ {recommendation}")
    elif final_score >= 0.70:
        st.info(f"ℹ️ {recommendation}")
    elif final_score >= 0.55:
        st.warning(f"⚠️ {recommendation}")
    else:
        st.error(f"❌ {recommendation}")


def main():
    """Main application"""
    
    # Display header
    display_header()
    
    # Sidebar configuration
    video_file, question, keywords = sidebar_config()
    
    # Analyze button
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("🚀 Analyze Interview", type="primary", use_container_width=True)
    
    # Main content area
    if analyze_button:
        if video_file is None:
            st.error("❌ Please upload a video file first!")
            return
        
        try:
            # Save uploaded file
            with st.spinner("Saving video file..."):
                video_path = save_uploaded_file(video_file)
            
            st.success(f"✅ Video uploaded: {video_file.name}")
            
            # Run analysis
            with st.spinner("🔄 Analyzing interview... This may take a few minutes."):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Step 1/5: Processing video...")
                progress_bar.progress(20)
                
                # Run pipeline
                results = analyze_interview(
                    video_path=str(video_path),
                    question=question,
                    expected_keywords=keywords
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
            
            # Display results
            display_results(results)
            
            # Download results option
            st.sidebar.markdown("---")
            st.sidebar.download_button(
                label="📥 Download Report (JSON)",
                data=json.dumps(results, indent=2),
                file_name=f"interview_report_{video_file.name}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions
        st.info("""
        ### 📋 How to Use:
        
        1. **Upload Video**: Upload a recorded interview video (MP4, AVI, MOV, etc.)
        2. **Enter Question**: Provide the interview question that was asked
        3. **Add Keywords** (Optional): Specify expected keywords for better evaluation
        4. **Click Analyze**: Press the "Analyze Interview" button to start processing
        5. **View Results**: Get comprehensive multimodal analysis with scores and feedback
        
        ### 🔒 Privacy & Security:
        - All processing is done locally
        - Videos are temporarily stored and deleted after analysis
        - No data is sent to external servers
        
        ### ⚡ System Requirements:
        - Supported formats: MP4, AVI, MOV, MKV, WebM
        - Maximum duration: 10 minutes
        - Clear audio and video quality recommended
        """)
        
        # Demo info
        st.markdown("---")
        st.markdown("### 🎬 Demo Mode")
        st.markdown("""
        **Try the system with your own interview videos!**
        
        For best results:
        - Ensure good lighting and clear audio
        - Position camera at eye level
        - Speak clearly towards the microphone
        - Keep the interview focused on the question
        """)


if __name__ == "__main__":
    main()

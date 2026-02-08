"""
Interview Analysis Web Application
Analyzes speech quality from interview responses using multi-dimensional analysis.
"""

import streamlit as st
import numpy as np
import os
import soundfile as sf
import tempfile
import sounddevice as sd
import json
from pathlib import Path

# Import analysis modules
try:
    from speech_analyzer import analyze_speech
    from confidence_scorer import analyze_and_score
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    st.error("⚠️ Analysis modules not found. Please check installation.")

st.set_page_config(page_title="Interview Speech Analyzer", layout="wide", page_icon="🎤")

# --------- WELCOME SCREEN ----------
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

if st.session_state.show_welcome:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');
        
        .welcome-container {
            text-align: center;
            padding: 80px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            margin: 20px 0;
        }
        .welcome-title {
            font-family: 'Poppins', sans-serif;
            font-size: 56px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .welcome-subtitle {
            font-size: 24px;
            color: #f0f0f0;
            margin-bottom: 10px;
        }
        .welcome-description {
            font-size: 18px;
            color: #e0e0e0;
            max-width: 600px;
            margin: 0 auto 30px;
            line-height: 1.6;
        }
        </style>

        <div class="welcome-container">
            <div class="welcome-title">🎤 Interview Speech Analyzer</div>
            <div class="welcome-subtitle">AI-Powered Speech Quality Analysis</div>
            <div class="welcome-description">
                Get comprehensive feedback on your interview responses with multi-dimensional 
                speech analysis including vocal quality, articulation, prosody, and timing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()

    st.stop()


# ---------- LOAD INTERVIEW QUESTIONS ----------
def load_interview_questions():
    """Load interview questions from JSON file."""
    try:
        questions_path = Path(__file__).parent / "interview_questions.json"
        with open(questions_path, 'r') as f:
            data = json.load(f)
            return data['questions']
    except Exception as e:
        st.error(f"Could not load interview questions: {e}")
        return []


# ---------- SILENCE CHECK ----------
def is_silent(path, threshold=0.01):
    """Return True if audio file is effectively silence."""
    try:
        import librosa
        y, sr = librosa.load(path, sr=None)
        if y.size == 0:
            return True
        rms = np.sqrt(np.mean(y**2))
        return rms < threshold
    except Exception:
        return False


# ---------- SESSION STATE ----------
if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = None

# ---------- MAIN UI ----------
st.title("🎤 Interview Speech Analyzer")
st.markdown("**Analyze your interview responses with AI-powered speech quality assessment**")

if not ANALYSIS_AVAILABLE:
    st.error("Analysis modules not available. Please check installation.")
    st.stop()

st.markdown("---")

# ========== INTERVIEW ANALYSIS ==========
st.header("📝 Interview Response Analysis")
st.info("💡 Select a question and record a 10-20 second natural response for comprehensive speech analysis")

# Load questions
questions = load_interview_questions()

if questions:
    # Question selection
    question_options = [f"{q['text']}" for q in questions]
    selected_question_idx = st.selectbox(
        "Choose an interview question:",
        range(len(question_options)),
        format_func=lambda i: question_options[i]
    )
    
    selected_question = questions[selected_question_idx]
    st.markdown(f"### 💬 Your Question:")
    st.info(f"**{selected_question['text']}**")
    st.caption(f"💡 {selected_question['description']}")
    st.caption(f"⏱️ Recommended duration: ~{selected_question['expected_duration']} seconds")
    
    # Recording options
    st.markdown("### 🎙️ Record Your Response")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**📁 Upload Audio File**\n\nUpload a pre-recorded response (WAV/MP3/M4A)")
    with col2:
        st.success("**🎤 Record Live**\n\nRecord your response directly in the browser")
    
    input_method = st.radio("Choose input method:", ["📁 Upload audio file", "🎤 Record live"], horizontal=True)
    
    audio_file = None
    if input_method == "📁 Upload audio file":
        audio_file = st.file_uploader("Upload your response", type=["wav", "mp3", "m4a"], key="upload")
    elif input_method == "🎤 Record live":
        duration = st.slider("Recording duration (seconds)", 10, 20, 15)
        if st.button(f"🎤 START RECORDING ({duration} seconds)", type="primary", key="record"):
            with st.spinner(f"🎤 Recording for {duration} seconds... Speak naturally!"):
                fs = 22050
                audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()
                st.session_state.recorded_audio_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ).name
                sf.write(st.session_state.recorded_audio_path, audio_data, fs)
                st.success(f"✅ {duration}-second recording saved!")
    
    # ANALYZE BUTTON
    if (audio_file is not None or st.session_state.recorded_audio_path) and st.button(
        "🎯 Analyze Speech Quality", type="primary", key="analyze", use_container_width=True
    ):
        temp_path = None
        try:
            with st.spinner("🎯 Performing comprehensive speech analysis..."):
                if audio_file:
                    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    with open(temp_path, "wb") as f:
                        f.write(audio_file.getvalue())
                else:
                    temp_path = st.session_state.recorded_audio_path
    
                if is_silent(temp_path):
                    st.warning("🎤 No voice detected. Please record again with clear speech.")
                    raise RuntimeError("Silent recording")
    
                # Perform comprehensive speech analysis
                features = analyze_speech(temp_path)
                
                if not features['success']:
                    st.error(f"❌ Analysis failed: {features.get('error', 'Unknown error')}")
                    raise RuntimeError("Analysis failed")
                
                # Calculate confidence scores and generate feedback
                results = analyze_and_score(features)
            
            st.success("✅ Analysis complete!")
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Speech Quality Analysis Results")
            
            # Overall score
            overall_score = results['confidence_scores']['overall_score']
            severity = results['confidence_scores']['severity']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Overall Speech Quality Score", f"{overall_score:.1f}/100")
            with col2:
                if severity == "Healthy":
                    st.success(f"✅ **Excellent Quality**")
                elif severity == "Mild Concern":
                    st.info(f"ℹ️ **Good Quality**")
                else:
                    st.warning(f"⚠️ **Needs Improvement**")
            
            # Progress bar for overall score
            st.progress(overall_score / 100)
            
            # Dimension scores
            st.markdown("### 📈 Detailed Feature Analysis")
            dim_scores = results['confidence_scores']['dimension_scores']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = dim_scores['vocal_quality']
                st.metric("🎵 Vocal Quality", f"{score:.1f}/100")
                st.progress(score / 100)
                if score >= 70:
                    st.caption("✅ Excellent")
                elif score >= 50:
                    st.caption("ℹ️ Good")
                else:
                    st.caption("⚠️ Needs work")
            
            with col2:
                score = dim_scores['articulation_clarity']
                st.metric("🗣️ Articulation", f"{score:.1f}/100")
                st.progress(score / 100)
                if score >= 70:
                    st.caption("✅ Excellent")
                elif score >= 50:
                    st.caption("ℹ️ Good")
                else:
                    st.caption("⚠️ Needs work")
            
            with col3:
                score = dim_scores['prosodic_variation']
                st.metric("🎭 Prosody", f"{score:.1f}/100")
                st.progress(score / 100)
                if score >= 70:
                    st.caption("✅ Excellent")
                elif score >= 50:
                    st.caption("ℹ️ Good")
                else:
                    st.caption("⚠️ Needs work")
            
            with col4:
                score = dim_scores['speech_timing']
                st.metric("⏱️ Timing", f"{score:.1f}/100")
                st.progress(score / 100)
                if score >= 70:
                    st.caption("✅ Excellent")
                elif score >= 50:
                    st.caption("ℹ️ Good")
                else:
                    st.caption("⚠️ Needs work")
            
            # Feedback section
            st.markdown("---")
            st.markdown("### 💡 Personalized Feedback")
            
            feedback = results['feedback']
            
            # Findings
            if feedback['findings']:
                st.markdown("#### 🔍 Key Observations")
                for finding in feedback['findings']:
                    st.info(f"• {finding}")
            
            # Recommendations
            if feedback['recommendations']:
                st.markdown("#### 🎯 Recommendations for Improvement")
                for rec in feedback['recommendations']:
                    st.success(f"• {rec}")
            
            # Summary
            st.markdown("---")
            st.markdown(f"**Summary:** {feedback['summary']}")
            
            # Download results option
            st.markdown("---")
            results_text = f"""
Interview Speech Analysis Results
=================================

Question: {selected_question['text']}
Overall Score: {overall_score:.1f}/100
Quality Level: {severity}

Detailed Scores:
- Vocal Quality: {dim_scores['vocal_quality']:.1f}/100
- Articulation Clarity: {dim_scores['articulation_clarity']:.1f}/100
- Prosodic Variation: {dim_scores['prosodic_variation']:.1f}/100
- Speech Timing: {dim_scores['speech_timing']:.1f}/100

Key Findings:
{chr(10).join('- ' + f for f in feedback['findings'])}

Recommendations:
{chr(10).join('- ' + r for r in feedback['recommendations'])}

Summary: {feedback['summary']}
"""
            st.download_button(
                label="📥 Download Results",
                data=results_text,
                file_name="interview_analysis_results.txt",
                mime="text/plain"
            )
            
        except RuntimeError as e:
            if "Silent recording" not in str(e):
                st.error(f"❌ Analysis failed: {str(e)}")
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
        finally:
            if temp_path and os.path.exists(temp_path) and temp_path != st.session_state.recorded_audio_path:
                os.remove(temp_path)
else:
    st.error("Could not load interview questions. Please check interview_questions.json file.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool analyzes your interview responses across multiple dimensions:
    
    - **Vocal Quality**: Pitch variation, voice stability
    - **Articulation**: Pronunciation clarity
    - **Prosody**: Intonation and rhythm
    - **Timing**: Speech rate and pauses
    
    Use this to practice and improve your interview skills!
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Score Guide")
    st.markdown("""
    - **70-100**: Excellent
    - **50-70**: Good
    - **30-50**: Fair
    - **0-30**: Needs improvement
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Speak naturally and clearly
    - Record in a quiet environment
    - Answer the question genuinely
    - Practice regularly to improve
    """)

st.markdown("---")
st.caption("🎤 Interview Speech Analyzer - Practice and improve your interview skills")

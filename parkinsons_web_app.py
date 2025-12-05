import streamlit as st
import numpy as np
import librosa
import joblib
import os
import soundfile as sf
import tempfile
import time
import sounddevice as sd

# ---------- EXTRACT FEATURES FUNCTION ----------
def extract_features(path, sr_target=22050, n_mfcc=20):
    y, sr = librosa.load(path, sr=sr_target)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    return np.hstack([mfcc_mean, zcr, rms, spec_centroid])

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = joblib.load("audio_parkinsons_model.pkl")
    scaler = joblib.load("audio_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------- UI ----------
st.set_page_config(page_title="Parkinson's Voice Screening", layout="wide")
st.title("🎤 Parkinson's Voice Screening Demo")
st.markdown("**Educational tool only - not medical diagnosis**")

# Symptoms sidebar (SIMPLE TERMS)
st.sidebar.header("📋 Symptom Questionnaire")
tremor = st.sidebar.checkbox("1. Hands shake when resting?")
slowness = st.sidebar.checkbox("2. Moving slower than before?")
stiffness = st.sidebar.checkbox("3. Muscles feel tight/stiff?")
balance = st.sidebar.checkbox("4. Feel unsteady or fall?")
voice_change = st.sidebar.checkbox("5. Voice quieter or flatter?")

symptom_score = sum([tremor, slowness, stiffness, balance, voice_change])
risk_level = "Low" if symptom_score <= 1 else "Medium" if symptom_score <= 3 else "High"
st.sidebar.metric("Symptom Risk", f"{symptom_score}/5", delta=f"→ {risk_level}")

# ---------- SESSION STATE ----------
if 'recorded_audio_path' not in st.session_state:
    st.session_state.recorded_audio_path = None

# Audio input
st.header("🔊 Voice Analysis")
input_method = st.radio("Choose input:", ["📁 Upload file", "🎙️ Record live (5 seconds)"])

audio_file = None

if input_method == "📁 Upload file":
    audio_file = st.file_uploader("Upload WAV/MP3/M4A", type=['wav', 'mp3', 'm4a'])

elif input_method == "🎙️ Record live (5 seconds)":
    if st.button("🎤 RECORD 5 SECONDS (Speak 'AH' now!)", type="primary"):
        with st.spinner("🎤 Recording... Speak 'AH' continuously!"):
            fs = 22050
            duration = 5
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            st.session_state.recorded_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            sf.write(st.session_state.recorded_audio_path, audio_data, fs)
            st.success("✅ 5-second recording saved!")

# ANALYZE BUTTON
if (audio_file is not None or st.session_state.recorded_audio_path) and st.button("🎯 Analyze Voice", type="primary"):
    temp_path = None
    
    try:
        with st.spinner("🎯 Analyzing voice features..."):
            # Handle file upload
            if audio_file:
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                with open(temp_path, "wb") as f:
                    f.write(audio_file.getvalue())
            else:
                temp_path = st.session_state.recorded_audio_path
            
            # Extract features & predict
            features = extract_features(temp_path)
            features_scaled = scaler.transform([features])
            probability_pd = model.predict_proba(features_scaled)[0, 1]
        
        # SIMPLIFIED 80% RULE ✅
        mode_text = "LIVE" if input_method == "🎙️ Record live (5 seconds)" else "FILE"
        
        # Results layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎙️ Parkinson's Probability", f"{probability_pd:.1%}")
        with col2:
            if probability_pd < 0.8:
                st.success("🟢 **YOU ARE HEALTHY**")
            else:
                st.error("🟡 **PARKINSON'S DETECTED**")
        with col3:
            st.metric("📋 Symptoms", risk_level)
        
        # Final recommendation (80% threshold only)
        st.markdown("---")
        if probability_pd < 0.8:
            st.success(f"✅ **YOU ARE HEALTHY** - {mode_text} voice analysis: **{probability_pd*100:.0f}%** Parkinson's probability")
            if symptom_score > 3:
                st.warning("⚠️ But high symptom score - consult doctor anyway")
        else:
            st.error(f"🚨 **PARKINSON'S DISEASE DETECTED** - {mode_text} voice analysis: **{probability_pd*100:.0f}%** (>80% threshold)")
            st.error("**Medical consultation strongly recommended**")
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path) and temp_path != st.session_state.recorded_audio_path:
            os.remove(temp_path)

# Footer
st.markdown("---")
st.caption("⚠️ Educational demo only - consult a doctor for diagnosis")

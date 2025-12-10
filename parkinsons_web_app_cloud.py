import streamlit as st
import numpy as np
import librosa
import joblib
import os
import soundfile as sf
import tempfile
from gtts import gTTS
import streamlit as st

st.set_page_config(page_title="Parkinson's Voice Screening", layout="wide")

# --------- WELCOME SCREEN USING SESSION STATE ----------
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

if st.session_state.show_welcome:
    st.markdown(
        """
        <style>
        .centered {
            text-align: center;
            padding-top: 120px;
        }
        .floating-title {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                         system-ui, sans-serif;
            font-size: 64px;
            font-weight: 800;
            letter-spacing: 6px;
            text-transform: uppercase;
            color: #ffffff;
            text-shadow: 0 0 20px rgba(0,0,0,0.6);
            animation: float 3s ease-in-out infinite;
        }
        .subtitle {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                         system-ui, sans-serif;
            font-size: 22px;
            color: #e0e0e0;
            margin-top: 30px;
        }
        @keyframes float {
            0%   { transform: translateY(0px); opacity: 0; }
            30%  { opacity: 1; }
            50%  { transform: translateY(-12px); }
            100% { transform: translateY(0px); opacity: 1; }
        }
        body {
            background: radial-gradient(circle at top, #4a90e2, #050816);
        }
        </style>
        <div class="centered">
            <div class="floating-title">WELCOME TO</div>
            <div class="subtitle">Parkinson's Voice Screening Demo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Enter app", type="primary", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()

    st.stop()  # do not run rest of app while welcome is shown

# ---------- EXTRACT FEATURES FUNCTION ----------
def extract_features(path, sr_target=22050, n_mfcc=20):
    y, sr = librosa.load(path, sr=sr_target)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    return np.hstack([mfcc_mean, zcr, rms, spec_centroid])

# ---------- SILENCE CHECK ----------
def is_silent(path, threshold=0.01):
    """Return True if audio file is effectively silence."""
    try:
        y, sr = librosa.load(path, sr=None)
        if y.size == 0:
            return True
        rms = np.sqrt(np.mean(y**2))
        return rms < threshold
    except Exception:
        return False

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = joblib.load("audio_parkinsons_model.pkl")
    scaler = joblib.load("audio_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------- LANGUAGE OPTIONS ----------
LANG_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
}

# ---------- BUILD SPOKEN TEXT ----------
def build_spoken_text(probability_pd, mode_text, risk_level, case_type, lang_label):
    p = int(round(probability_pd * 100))

    translations = {
        "English": {
            "low":    f"Symptoms indicate low risk. {mode_text} voice shows {p} percent Parkinson's probability. This is an educational demo only, not a medical diagnosis.",
            "medium": f"Symptoms indicate medium risk. {mode_text} voice shows {p} percent Parkinson's probability. Please monitor symptoms and consult a doctor if needed. This is an educational demo only, not a medical diagnosis.",
            "high":   f"Symptoms indicate high risk. {mode_text} voice shows {p} percent Parkinson's probability. Medical consultation is strongly recommended. This is an educational demo only, not a medical diagnosis.",
        },
        "Hindi": {
            "low":    f"Lakshan kam jokhim dikha rahe hain. {mode_text} awaaz ke aadhar par {p} pratishat Parkinson sambhavana hai. Yeh sirf shiksha ke liye demo hai.",
            "medium": f"Lakshan madhyam jokhim dikha rahe hain. {mode_text} awaaz ke aadhar par {p} pratishat Parkinson sambhavana hai. Kripya lakshanon par dhyan rakhen aur zarurat par doctor se salah len.",
            "high":   f"Lakshan zyada jokhim dikha rahe hain. {mode_text} awaaz ke aadhar par {p} pratishat Parkinson sambhavana hai. Doctor se turant salah len.",
        },
        "Bengali": {
            "low":    f"Lakshan kom jhuki dekhachhe. {mode_text} voice e {p} shotangsho Parkinson sambhabona.",
            "medium": f"Lakshan moddhom jhuki dekhachhe. {mode_text} voice e {p} shotangsho Parkinson sambhabona.",
            "high":   f"Lakshan beshi jhuki dekhachhe. {mode_text} voice e {p} shotangsho Parkinson sambhabona. Doctor er kache jaan.",
        },
        "Telugu": {
            "low":    f"Lakshanalu takkuva risk chupistunnayi. {mode_text} voice dvara {p} shatam Parkinson sambhavana.",
            "medium": f"Lakshanalu madhyama risk chupistunnayi. {mode_text} voice dvara {p} shatam Parkinson sambhavana.",
            "high":   f"Lakshanalu ekkuva risk chupistunnayi. {mode_text} voice dvara {p} shatam Parkinson sambhavana. Dayachesi doctor ni kalavandi.",
        },
        "Tamil": {
            "low":    f"Lakshanangal kuraindha abaya nilaiyai kaattukinrana. {mode_text} voice il {p} sathavitham Parkinson saathiyam.",
            "medium": f"Lakshanangal madhyama abaya nilaiyai kaattukinrana. {mode_text} voice il {p} sathavitham Parkinson saathiyam.",
            "high":   f"Lakshanangal adhiga abaya nilaiyai kaattukinrana. {mode_text} voice il {p} sathavitham Parkinson saathiyam. Doctor ai sandhithal nallathu.",
        },
        "Kannada": {
            "low":    f"Lakshanagalu kadime apaya torisuttave. {mode_text} voice nalli {p} shatam Parkinson sambhavane.",
            "medium": f"Lakshanagalu madhyama apaya torisuttave. {mode_text} voice nalli {p} shatam Parkinson sambhavane.",
            "high":   f"Lakshanagalu hecchu apaya torisuttave. {mode_text} voice nalli {p} shatam Parkinson sambhavane. Dayavittu doctorannu bheti madi.",
        },
        "Malayalam": {
            "low":    f"Lakshanangal kuranja risk kaanikkunnu. {mode_text} voice il {p} shathamanam Parkinson sambhavana.",
            "medium": f"Lakshanangal madhyama risk kaanikkunnu. {mode_text} voice il {p} shathamanam Parkinson sambhavana.",
            "high":   f"Lakshanangal valare koodiya risk kaanikkunnu. {mode_text} voice il {p} shathamanam Parkinson sambhavana. Dayavaayi doctor ne kaananam.",
        },
        "Gujarati": {
            "low":    f"Lakshano ochha jokham darshave chhe. {mode_text} voice pramane {p} takka Parkinson ni sambhavna chhe.",
            "medium": f"Lakshano madhyam jokham darshave chhe. {mode_text} voice pramane {p} takka Parkinson ni sambhavna chhe.",
            "high":   f"Lakshano vadhare jokham darshave chhe. {mode_text} voice pramane {p} takka Parkinson ni sambhavna chhe. Krupaya doctor ne bataavo.",
        },
        "Marathi": {
            "low":    f"Lakshane kami dhoka darshavtat. {mode_text} voice madhye {p} takke Parkinson chi shak्यता aahe.",
            "medium": f"Lakshane madhyam dhoka darshavtat. {mode_text} voice madhye {p} takke Parkinson chi shak्यता aahe.",
            "high":   f"Lakshane mothya dhokyache suchak aahet. {mode_text} voice madhye {p} takke Parkinson chi shak्यता aahe. Krupaya doctor la bhet dya.",
        },
        "Punjabi": {
            "low":    f"Lakshan ghatt khatra dikhaunde ne. {mode_text} voice vich {p} percent Parkinson di sambhavna.",
            "medium": f"Lakshan darmiyani khatra dikhaunde ne. {mode_text} voice vich {p} percent Parkinson di sambhavna.",
            "high":   f"Lakshan vadda khatra dikhaunde ne. {mode_text} voice vich {p} percent Parkinson di sambhavna. Doctor nu jaroor dikhao.",
        },
        "Urdu": {
            "low":    f"Alamat kam khatra dikhati hain. {mode_text} awaaz se {p} feesad Parkinson ka imkaan hai.",
            "medium": f"Alamat darmiyani khatra dikhati hain. {mode_text} awaaz se {p} feesad Parkinson ka imkaan hai.",
            "high":   f"Alamat zyada khatra dikhati hain. {mode_text} awaaz se {p} feesad Parkinson ka imkaan hai. Barah-e-karam doctor se raabta karein.",
        },
        "Spanish": {
            "low":    f"Los síntomas indican bajo riesgo. La voz {mode_text} muestra un {p} por ciento de probabilidad de Parkinson.",
            "medium": f"Los síntomas indican riesgo medio. La voz {mode_text} muestra un {p} por ciento de probabilidad de Parkinson.",
            "high":   f"Los síntomas indican alto riesgo. La voz {mode_text} muestra un {p} por ciento de probabilidad de Parkinson. Consulte a un médico.",
        },
        "French": {
            "low":    f"Les symptômes indiquent un faible risque. La voix {mode_text} montre {p} pour cent de probabilité de Parkinson.",
            "medium": f"Les symptômes indiquent un risque moyen. La voix {mode_text} montre {p} pour cent de probabilité de Parkinson.",
            "high":   f"Les symptômes indiquent un risque élevé. La voix {mode_text} montre {p} pour cent de probabilité de Parkinson. Consultez un médecin.",
        },
        "German": {
            "low":    f"Die Symptome weisen auf ein geringes Risiko hin. Die {mode_text}-Stimme zeigt {p} Prozent Wahrscheinlichkeit für Parkinson.",
            "medium": f"Die Symptome weisen auf ein mittleres Risiko hin. Die {mode_text}-Stimme zeigt {p} Prozent Wahrscheinlichkeit für Parkinson.",
            "high":   f"Die Symptome weisen auf ein hohes Risiko hin. Die {mode_text}-Stimme zeigt {p} Prozent Wahrscheinlichkeit für Parkinson. Bitte suchen Sie einen Arzt auf.",
        },
        "Italian": {
            "low":    f"I sintomi indicano un rischio basso. La voce {mode_text} mostra una probabilità di Parkinson del {p} per cento.",
            "medium": f"I sintomi indicano un rischio medio. La voce {mode_text} mostra una probabilità di Parkinson del {p} per cento.",
            "high":   f"I sintomi indicano un rischio alto. La voce {mode_text} mostra una probabilità di Parkinson del {p} per cento. Consultare un medico.",
        },
        "Portuguese": {
            "low":    f"Os sintomas indicam baixo risco. A voz {mode_text} mostra {p} por cento de probabilidade de Parkinson.",
            "medium": f"Os sintomas indicam risco médio. A voz {mode_text} mostra {p} por cento de probabilidade de Parkinson.",
            "high":   f"Os sintomas indicam alto risco. A voz {mode_text} mostra {p} por cento de probabilidade de Parkinson. Consulte um médico.",
        },
    }

    risk_key = "low" if case_type == "low" else "high" if case_type == "high" else "medium"
    return translations.get(lang_label, translations["English"])[risk_key]

# ---------- TTS HELPER ----------
def speak_result(text: str, lang_code: str = "en"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(tmp.name)
            tmp.seek(0)
            audio_bytes = tmp.read()
        os.remove(tmp.name)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning(f"🔈 Could not generate speech: {e}")

# ---------- UI ----------
st.set_page_config(page_title="Parkinson's Voice Screening", layout="wide")
st.title("🎤 Parkinson's Voice Screening Demo")
st.markdown("**Educational tool only - not medical diagnosis**")

# Symptoms sidebar
st.sidebar.header("📋 Symptom Questionnaire")
tremor = st.sidebar.checkbox("1. Hands shake when resting?")
slowness = st.sidebar.checkbox("2. Moving slower than before?")
stiffness = st.sidebar.checkbox("3. Muscles feel tight/stiff?")
balance = st.sidebar.checkbox("4. Feel unsteady or fall?")
voice_change = st.sidebar.checkbox("5. Voice quieter or flatter?")

symptom_score = sum([tremor, slowness, stiffness, balance, voice_change])
risk_level = "Low" if symptom_score <= 1 else "Medium" if symptom_score <= 3 else "High"
st.sidebar.metric("Symptom Risk", f"{symptom_score}/5", delta=f"→ {risk_level}")

# Language selection
st.sidebar.markdown("### 🔊 Spoken result language")
lang_label = st.sidebar.selectbox("Choose language", list(LANG_OPTIONS.keys()), index=0)
tts_lang_code = LANG_OPTIONS[lang_label]

# ---------- AUDIO INPUT (UPLOAD ONLY) ----------
st.header("🔊 Voice Analysis")
st.write("On the web version, please upload a 5‑second 'AH' recording (no live mic support).")
audio_file = st.file_uploader("Upload WAV/MP3/M4A", type=["wav", "mp3", "m4a"])

# ANALYZE BUTTON
if audio_file is not None and st.button("🎯 Analyze Voice", type="primary"):
    temp_path = None
    try:
        with st.spinner("🎯 Analyzing voice features..."):
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(temp_path, "wb") as f:
                f.write(audio_file.getvalue())

            if is_silent(temp_path):
                st.warning("🎤 It seems the file has almost no voice. Please upload a clear 5‑second 'AH' recording.")
                raise RuntimeError("Silent recording")

            features = extract_features(temp_path)
            features_scaled = scaler.transform([features])
            probability_pd = model.predict_proba(features_scaled)[0, 1]

        mode_text = "file"

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

        if probability_pd < 0.8:
            case_type = "low"
            st.markdown("---")
            st.success(
                f"✅ **YOU ARE HEALTHY** - FILE voice analysis: "
                f"**{probability_pd*100:.0f}%** Parkinson's probability"
            )
            if symptom_score > 3:
                st.warning("⚠️ But high symptom score - consult a doctor anyway.")
        elif probability_pd >= 0.8:
            case_type = "high"
            st.markdown("---")
            st.error(
                f"🚨 **PARKINSON'S DISEASE DETECTED** - FILE voice analysis: "
                f"**{probability_pd*100:.0f}%** (>80% threshold)"
            )
            st.error("**Medical consultation strongly recommended.**")
        else:
            case_type = "medium"
            st.markdown("---")
            st.warning(
                f"⚠️ Mixed result - FILE voice analysis: "
                f"**{probability_pd*100:.0f}%** Parkinson's probability"
            )

        spoken_text = build_spoken_text(
            probability_pd, mode_text, risk_level, case_type, lang_label
        )

        st.markdown("### 🗣 Spoken result")
        speak_result(spoken_text, lang_code=tts_lang_code)

    except RuntimeError as e:
        if "Silent recording" not in str(e):
            st.error(f"❌ Analysis failed: {str(e)}")
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

st.markdown("---")
st.caption("⚠️ Educational demo only - consult a doctor for diagnosis")

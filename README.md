# 🎤 Parkinson's Voice Screening (Educational Demo)

This project is a **Streamlit web app and CLI tool** that uses voice features and a simple questionnaire to estimate the probability of Parkinson's disease from sustained “AH” sounds.  
It is **for education and practice only – not for real medical diagnosis**.

## 🚀 Features

- 🎙️ Voice analysis from:
  - Live 5‑second microphone recording
  - Uploaded WAV/MP3/M4A files
- 🧠 Machine‑learning model (Random Forest) trained on voice features
- 📋 Symptom questionnaire with simple, patient‑friendly language
- 🖥️ Web app (Streamlit) and terminal/CLI version
- 🔊 Optional text‑to‑speech feedback in the CLI app

## 🛠️ Tech Stack

- Python, NumPy, pandas
- Librosa (audio feature extraction)
- scikit‑learn (RandomForestClassifier)
- Streamlit (web UI)
- sounddevice, soundfile, gTTS, pygame (audio I/O and TTS)

## 📦 Setup (Local)

git clone https://github.com/luckybiswal98210-eng/-parkinsons-voice-screening-.git cd -parkinsons-voice-screening-
python3 -m venv venv        # Windows: python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt


Place your trained model files in the project folder:

- `audio_parkinsons_model.pkl`
- `audio_scaler.pkl`

## ▶️ Run the Apps

### Web app (recommended)

streamlit run parkinsons_web_app.py

Then open the browser at `http://localhost:8501` if it doesn’t open automatically.

### CLI app

python parkinsons_audio_app.py

Follow the on‑screen instructions to answer symptom questions and provide audio.

## ⚠️ Disclaimer

This project is **for educational and research practice only**.  
It **must not** be used for real medical diagnosis or treatment decisions.  
Always consult a qualified doctor for any health concerns.

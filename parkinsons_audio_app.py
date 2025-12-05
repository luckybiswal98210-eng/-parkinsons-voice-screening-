import os
import numpy as np
import librosa
import joblib
from gtts import gTTS
import pygame
import tempfile
import sounddevice as sd
import soundfile as sf

# Initialize pygame mixer for audio playback
pygame.mixer.init()

model = joblib.load("audio_parkinsons_model.pkl")
scaler = joblib.load("audio_scaler.pkl")

def extract_features(path, sr_target=22050, n_mfcc=20):
    y, sr = librosa.load(path, sr=sr_target)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    return np.hstack([mfcc_mean, zcr, rms, spec_centroid])

def speak(text):
    print(f"🗣️  {text}")
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            tts.save(tmp.name)
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        os.unlink(tmp.name)
    except Exception as e:
        print(f"❌ TTS failed: {e}")

def ask_yes_no(question):
    while True:
        ans = input(question + " (y/n): ").strip().lower()
        if ans in ['y', 'yes']: return 1
        if ans in ['n', 'no']: return 0
        print("Please answer y or n")

def record_audio(duration=5, fs=22050):
    print(f"🎤 Recording for {duration} seconds... Speak 'AH' continuously!")
    print("🔴 Recording... (speak now)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("✅ Recording complete!")
    return audio.flatten()

print("🎤 Parkinson's Voice Screening Demo\n")
print("Choose mode: 1) File upload  2) Live microphone")

mode = input("Enter 1 or 2: ").strip()
print()

# Symptoms
symptoms = {}
symptoms["tremor"] = ask_yes_no("1. Do your hands shake when resting?")
symptoms["slowness"] = ask_yes_no("2. Do you move slower than before?")
symptoms["stiffness"] = ask_yes_no("3. Do your muscles feel tight/stiff?")
symptoms["balance"] = ask_yes_no("4. Do you feel unsteady or fall?")
symptoms["voice"] = ask_yes_no("5. Is your voice quieter or flatter?")

symptom_score = sum(symptoms.values())
risk_level = "Low" if symptom_score <= 1 else "Medium" if symptom_score <= 3 else "High"
print(f"\n📋 Symptom risk: {symptom_score}/5 → {risk_level}")

# Get audio path
if mode == "1":
    print("\n📁 FILE MODE")
    while True:
        audio_path = input("Full path: ").strip()
        if os.path.exists(audio_path):
            analysis_path = audio_path
            break
        print("❌ File not found. Try again.")

elif mode == "2":
    print("\n🎙️  LIVE MODE")
    input("Press Enter when ready to record...")
    audio_data = record_audio(duration=5)
    analysis_path = "temp_mic_recording.wav"
    sf.write(analysis_path, audio_data, 22050)
else:
    print("❌ Invalid choice. Exiting.")
    exit()

# ANALYSIS
print("🎯 Analyzing voice...")
result_text = "Analysis failed."
try:
    features = extract_features(analysis_path)
    features_scaled = scaler.transform([features])
    probability_pd = model.predict_proba(features_scaled)[0, 1]
    prediction = model.predict(features_scaled)[0]
    
    print("\n" + "="*50)
    mode_text = "LIVE" if mode == "2" else "FILE"
    print(f"🎙️  {mode_text} Voice model: {probability_pd:.1%} Parkinson's probability")
    prediction_text = "Parkinson's" if prediction == 1 else "Healthy"
    print(f"📊 Prediction: {prediction_text}")
    print(f"📋 Symptoms: {risk_level}")

    if risk_level == "High" and prediction == 1 and probability_pd > 0.8:
        result_text = f"Symptoms indicate high risk. {mode_text.lower()} voice shows {probability_pd*100:.0f} percent Parkinson's probability. Medical consultation strongly recommended. This is for educational demo not for medical diagnosis."
        print("🚨 HIGH overall risk")
    elif risk_level == "Low" and prediction == 0:
        result_text = f"Symptoms indicate low risk. {mode_text.lower()} voice shows healthy result. This is for educational demo not for medical diagnosis."
        print("✅ LOW overall risk")
    else:
        result_text = f"Symptoms indicate {risk_level.lower()} risk. {mode_text.lower()} voice shows {probability_pd*100:.0f} percent Parkinson's probability. Monitor symptoms. This is for educational demo not for medical diagnosis."
        print("⚠️  Mixed results")

except Exception as e:
    print(f"❌ Analysis failed: {e}")
    result_text = "Analysis failed. Please try again."

# Speak ONLY combined result
print("\n🗣️  Speaking final combined result only...")
speak(result_text)

# Cleanup
if mode == "2" and os.path.exists("temp_mic_recording.wav"):
    os.remove("temp_mic_recording.wav")

print("\n⚠️  Educational demo only - not medical diagnosis!")

"""
Advanced speech analysis module for interview response quality assessment.
Extracts vocal, articulatory, prosodic, and timing features from conversational speech.
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


def extract_vocal_features(audio_path, sr=22050):
    """
    Extract vocal quality features including pitch, jitter, shimmer, and HNR.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        dict: Vocal features
    """
    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Load with Parselmouth for advanced voice analysis
    try:
        sound = parselmouth.Sound(audio_path)
        
        # Extract pitch
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values != 0]  # Remove unvoiced frames
        
        # Pitch statistics
        f0_mean = np.mean(f0_values) if len(f0_values) > 0 else 0
        f0_std = np.std(f0_values) if len(f0_values) > 0 else 0
        f0_min = np.min(f0_values) if len(f0_values) > 0 else 0
        f0_max = np.max(f0_values) if len(f0_values) > 0 else 0
        f0_range = f0_max - f0_min
        
        # Jitter (pitch perturbation)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Shimmer (amplitude perturbation)
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Harmonics-to-Noise Ratio
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
    except Exception as e:
        print(f"Parselmouth analysis failed: {e}, using librosa fallback")
        # Fallback to librosa-only analysis
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600, sr=sr)
        f0_values = f0[~np.isnan(f0)]
        
        f0_mean = np.mean(f0_values) if len(f0_values) > 0 else 0
        f0_std = np.std(f0_values) if len(f0_values) > 0 else 0
        f0_min = np.min(f0_values) if len(f0_values) > 0 else 0
        f0_max = np.max(f0_values) if len(f0_values) > 0 else 0
        f0_range = f0_max - f0_min
        jitter = 0
        shimmer = 0
        hnr = 0
    
    # RMS energy (loudness)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    return {
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'f0_min': f0_min,
        'f0_max': f0_max,
        'f0_range': f0_range,
        'jitter': jitter,
        'shimmer': shimmer,
        'hnr': hnr,
        'rms_mean': rms_mean,
        'rms_std': rms_std
    }


def extract_articulatory_features(audio_path, sr=22050):
    """
    Extract articulation features including formants, vowel space, and clarity measures.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        dict: Articulatory features
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Formant analysis (approximation using spectral features)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # Zero crossing rate (articulation clarity indicator)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Spectral contrast (articulation precision)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast)
    
    return {
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std,
        'spectral_contrast_mean': spectral_contrast_mean
    }


def extract_prosodic_features(audio_path, sr=22050):
    """
    Extract prosodic features including intonation, rhythm, and stress patterns.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        dict: Prosodic features
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Pitch variation (already computed in vocal features, but important for prosody)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600, sr=sr)
    f0_values = f0[~np.isnan(f0)]
    
    # Pitch variation coefficient
    pitch_variation = np.std(f0_values) / np.mean(f0_values) if len(f0_values) > 0 and np.mean(f0_values) > 0 else 0
    
    # Energy variation (stress patterns)
    rms = librosa.feature.rms(y=y)[0]
    energy_variation = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
    
    # Rhythm features using onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # Mel-frequency cepstral coefficients (speech quality)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    return {
        'pitch_variation': pitch_variation,
        'energy_variation': energy_variation,
        'tempo': tempo,
        'mfcc_mean': mfcc_mean.tolist(),
        'mfcc_std': mfcc_std.tolist()
    }


def extract_timing_features(audio_path, sr=22050, silence_threshold=0.01):
    """
    Extract timing features including pauses, speech rate, and articulation rate.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        silence_threshold: Threshold for silence detection
        
    Returns:
        dict: Timing features
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Total duration
    total_duration = len(y) / sr
    
    # Detect speech/silence using energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_threshold = silence_threshold
    
    # Speech frames
    speech_frames = rms > rms_threshold
    
    # Calculate speech and pause durations
    frame_duration = 512 / sr  # hop_length / sr
    speech_duration = np.sum(speech_frames) * frame_duration
    pause_duration = total_duration - speech_duration
    
    # Pause frequency (number of pauses per second)
    # Detect transitions from speech to silence
    transitions = np.diff(speech_frames.astype(int))
    num_pauses = np.sum(transitions == -1)
    pause_frequency = num_pauses / total_duration if total_duration > 0 else 0
    
    # Speech rate (approximate syllables per second)
    # Using onset detection as proxy for syllables
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    num_syllables = len(onsets)
    speech_rate = num_syllables / speech_duration if speech_duration > 0 else 0
    
    # Articulation rate (syllables per second of actual speech, excluding pauses)
    articulation_rate = num_syllables / speech_duration if speech_duration > 0 else 0
    
    # Pause-to-speech ratio
    pause_to_speech_ratio = pause_duration / speech_duration if speech_duration > 0 else 0
    
    return {
        'total_duration': total_duration,
        'speech_duration': speech_duration,
        'pause_duration': pause_duration,
        'pause_frequency': pause_frequency,
        'num_pauses': num_pauses,
        'speech_rate': speech_rate,
        'articulation_rate': articulation_rate,
        'pause_to_speech_ratio': pause_to_speech_ratio
    }


def analyze_speech(audio_path, sr=22050):
    """
    Comprehensive speech analysis combining all feature types.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        dict: All extracted features organized by category
    """
    try:
        vocal_features = extract_vocal_features(audio_path, sr)
        articulatory_features = extract_articulatory_features(audio_path, sr)
        prosodic_features = extract_prosodic_features(audio_path, sr)
        timing_features = extract_timing_features(audio_path, sr)
        
        return {
            'vocal': vocal_features,
            'articulatory': articulatory_features,
            'prosodic': prosodic_features,
            'timing': timing_features,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

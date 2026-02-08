"""
Confidence scoring and feedback generation for interview speech quality assessment.
Analyzes speech features and provides actionable feedback for interview performance.
"""

import json
import numpy as np
from pathlib import Path


def load_config():
    """Load configuration file with baseline ranges and weights."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def normalize_feature(value, baseline_range, inverse=False):
    """
    Normalize a feature value to 0-1 scale based on baseline range.
    
    Args:
        value: Feature value
        baseline_range: [min, max] healthy range
        inverse: If True, lower values are better
        
    Returns:
        float: Normalized score (0-1, where 1 is healthy)
    """
    if value == 0:
        return 0.5  # Neutral score for missing values
    
    min_val, max_val = baseline_range
    
    if inverse:
        # For features where lower is better (e.g., jitter, shimmer)
        if value <= min_val:
            return 1.0
        elif value >= max_val:
            return 0.0
        else:
            return 1.0 - (value - min_val) / (max_val - min_val)
    else:
        # For features where being in range is better
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            return max(0.0, value / min_val)
        else:
            return max(0.0, 1.0 - (value - max_val) / max_val)


def calculate_vocal_score(vocal_features, config):
    """Calculate vocal quality score (0-100)."""
    baseline = config['baseline_ranges']
    
    scores = []
    
    # F0 variability (std) - healthy range
    if vocal_features['f0_std'] > 0:
        f0_std_score = normalize_feature(vocal_features['f0_std'], baseline['f0_std'])
        scores.append(f0_std_score)
    
    # Jitter - lower is better
    if vocal_features['jitter'] > 0:
        jitter_score = normalize_feature(vocal_features['jitter'], baseline['jitter'], inverse=True)
        scores.append(jitter_score)
    
    # Shimmer - lower is better
    if vocal_features['shimmer'] > 0:
        shimmer_score = normalize_feature(vocal_features['shimmer'], baseline['shimmer'], inverse=True)
        scores.append(shimmer_score)
    
    # HNR - higher is better
    if vocal_features['hnr'] > 0:
        hnr_score = normalize_feature(vocal_features['hnr'], baseline['hnr'])
        scores.append(hnr_score)
    
    # Pitch range
    if vocal_features['f0_range'] > 0:
        pitch_range_score = normalize_feature(vocal_features['f0_range'], baseline['pitch_range'])
        scores.append(pitch_range_score)
    
    return np.mean(scores) * 100 if scores else 50.0


def calculate_articulatory_score(articulatory_features, config):
    """Calculate articulation clarity score (0-100)."""
    # Use spectral features as proxies for articulation quality
    # Higher spectral contrast and centroid variation indicate clearer articulation
    
    scores = []
    
    # Spectral contrast (higher is better for clarity)
    if articulatory_features['spectral_contrast_mean'] > 15:
        scores.append(1.0)
    elif articulatory_features['spectral_contrast_mean'] > 10:
        scores.append(0.7)
    else:
        scores.append(0.4)
    
    # ZCR variation (moderate variation is good)
    if 0.05 < articulatory_features['zcr_mean'] < 0.15:
        scores.append(1.0)
    else:
        scores.append(0.6)
    
    return np.mean(scores) * 100 if scores else 50.0


def calculate_prosodic_score(prosodic_features, config):
    """Calculate prosodic variation score (0-100)."""
    scores = []
    
    # Pitch variation (coefficient of variation)
    if prosodic_features['pitch_variation'] > 0.15:
        scores.append(1.0)
    elif prosodic_features['pitch_variation'] > 0.08:
        scores.append(0.7)
    else:
        scores.append(0.3)
    
    # Energy variation
    if prosodic_features['energy_variation'] > 0.3:
        scores.append(1.0)
    elif prosodic_features['energy_variation'] > 0.15:
        scores.append(0.7)
    else:
        scores.append(0.4)
    
    return np.mean(scores) * 100 if scores else 50.0


def calculate_timing_score(timing_features, config):
    """Calculate speech timing score (0-100)."""
    baseline = config['baseline_ranges']
    
    scores = []
    
    # Speech rate
    speech_rate_score = normalize_feature(timing_features['speech_rate'], baseline['speech_rate'])
    scores.append(speech_rate_score)
    
    # Pause frequency
    pause_freq_score = normalize_feature(timing_features['pause_frequency'], baseline['pause_frequency'])
    scores.append(pause_freq_score)
    
    # Pause-to-speech ratio (lower is generally better)
    if timing_features['pause_to_speech_ratio'] < 0.3:
        scores.append(1.0)
    elif timing_features['pause_to_speech_ratio'] < 0.5:
        scores.append(0.7)
    else:
        scores.append(0.4)
    
    return np.mean(scores) * 100 if scores else 50.0


def calculate_confidence_score(features_dict):
    """
    Calculate overall confidence score and dimension scores.
    
    Args:
        features_dict: Dictionary with 'vocal', 'articulatory', 'prosodic', 'timing' features
        
    Returns:
        dict: Confidence scores and dimension breakdown
    """
    config = load_config()
    weights = config['confidence_weights']
    
    # Calculate dimension scores
    vocal_score = calculate_vocal_score(features_dict['vocal'], config)
    articulatory_score = calculate_articulatory_score(features_dict['articulatory'], config)
    prosodic_score = calculate_prosodic_score(features_dict['prosodic'], config)
    timing_score = calculate_timing_score(features_dict['timing'], config)
    
    # Calculate weighted overall score
    overall_score = (
        vocal_score * weights['vocal'] +
        articulatory_score * weights['articulatory'] +
        prosodic_score * weights['prosodic'] +
        timing_score * weights['timing']
    )
    
    # Determine quality level
    severity_thresholds = config['severity_thresholds']
    if overall_score >= severity_thresholds['mild'] * 100:
        severity = "Excellent"
    elif overall_score >= severity_thresholds['moderate'] * 100:
        severity = "Good"
    elif overall_score >= severity_thresholds['severe'] * 100:
        severity = "Fair"
    else:
        severity = "Needs Improvement"
    
    return {
        'overall_score': overall_score,
        'severity': severity,
        'dimension_scores': {
            'vocal_quality': vocal_score,
            'articulation_clarity': articulatory_score,
            'prosodic_variation': prosodic_score,
            'speech_timing': timing_score
        }
    }


def generate_feedback(features_dict, confidence_scores):
    """
    Generate personalized feedback based on feature analysis.
    
    Args:
        features_dict: Dictionary with extracted features
        confidence_scores: Dictionary with confidence scores
        
    Returns:
        dict: Feedback with findings and recommendations
    """
    findings = []
    recommendations = []
    
    vocal = features_dict['vocal']
    prosodic = features_dict['prosodic']
    timing = features_dict['timing']
    scores = confidence_scores['dimension_scores']
    
    # Vocal quality feedback
    if scores['vocal_quality'] < 60:
        if vocal['f0_std'] < 20:
            findings.append("Limited pitch variation detected (monotone speech)")
            recommendations.append("Practice varying your pitch - try reading with more expression")
        if vocal['jitter'] > 0.01:
            findings.append("Voice instability detected")
            recommendations.append("Practice vocal exercises to improve voice stability")
        if vocal['hnr'] < 10:
            findings.append("Voice clarity could be improved")
            recommendations.append("Practice breath support and speak with more vocal energy")
    else:
        if vocal['f0_std'] > 30:
            findings.append("Good pitch variation - your speech is expressive")
    
    # Articulation feedback
    if scores['articulation_clarity'] < 60:
        findings.append("Articulation could be clearer")
        recommendations.append("Practice speaking slowly and clearly, emphasizing consonants")
        recommendations.append("Try tongue twisters and articulation exercises")
    else:
        findings.append("Clear articulation - words are well-pronounced")
    
    # Prosodic feedback
    if scores['prosodic_variation'] < 60:
        if prosodic['pitch_variation'] < 0.08:
            findings.append("Limited intonation variation")
            recommendations.append("Practice speaking with more emotional expression")
        if prosodic['energy_variation'] < 0.15:
            findings.append("Speech energy could be more varied")
            recommendations.append("Try emphasizing important words and varying your volume")
    else:
        findings.append("Good prosodic variation - natural and engaging speech")
    
    # Timing feedback
    if scores['speech_timing'] < 60:
        if timing['speech_rate'] < 4.0:
            findings.append("Speech rate is slower than average")
            recommendations.append("Practice speaking at a comfortable but steady pace")
        elif timing['speech_rate'] > 6.5:
            findings.append("Speech rate is faster than average")
            recommendations.append("Try slowing down and pausing between thoughts")
        if timing['pause_frequency'] > 0.3:
            findings.append("Frequent pauses detected")
            recommendations.append("Practice continuous speech with fewer interruptions")
    else:
        findings.append("Good speech timing and pacing")
    
    # If everything is good
    if not findings:
        findings.append("Excellent speech quality across all dimensions")
        recommendations.append("Continue practicing to maintain your skills")
        recommendations.append("Try more challenging questions to further improve")
    
    return {
        'findings': findings,
        'recommendations': recommendations,
        'summary': f"Analysis complete. Speech quality: {confidence_scores['severity']}. "
                   f"Overall score: {confidence_scores['overall_score']:.1f}/100."
    }


def analyze_and_score(features_dict):
    """
    Complete analysis pipeline: calculate scores and generate feedback.
    
    Args:
        features_dict: Dictionary with extracted features
        
    Returns:
        dict: Complete analysis results
    """
    confidence_scores = calculate_confidence_score(features_dict)
    feedback = generate_feedback(features_dict, confidence_scores)
    
    return {
        'confidence_scores': confidence_scores,
        'feedback': feedback,
        'features': features_dict
    }

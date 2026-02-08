# 🎤 Interview Analysis App

AI-powered speech quality analysis tool for interview practice and improvement.

## 🚀 Features

- **🎙️ Multi-dimensional Speech Analysis**
  - Vocal Quality (pitch variation, voice stability, harmonics)
  - Articulation Clarity (pronunciation, speech clarity)
  - Prosodic Variation (intonation, rhythm, expression)
  - Speech Timing (pauses, speech rate, pacing)

- **📝 Professional Interview Questions**
  - Common interview questions across multiple categories
  - Personal introduction, behavioral, motivation, and career goals
  - Recommended response durations (10-20 seconds)

- **💡 Personalized Feedback**
  - Detailed analysis with specific observations
  - Actionable recommendations for improvement
  - Overall quality scores (0-100) for each dimension

- **🎯 Flexible Recording Options**
  - Upload pre-recorded audio (WAV/MP3/M4A)
  - Record live directly in the browser
  - Adjustable recording duration

## 🛠️ Tech Stack

- **Python** - Core language
- **Streamlit** - Web UI framework
- **Librosa** - Audio feature extraction
- **Parselmouth/Praat** - Advanced voice analysis
- **NumPy** - Numerical computations
- **SoundDevice/SoundFile** - Audio I/O

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/InterviewAnalysisApp.git
   cd InterviewAnalysisApp
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

### Run the Web App

```bash
streamlit run interview_analyzer_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### How to Use

1. **Welcome Screen** - Click "Start Analysis" to begin
2. **Select Question** - Choose from 8 professional interview questions
3. **Record Response** - Either upload audio or record live (10-20 seconds)
4. **Analyze** - Click "Analyze Speech Quality" to get your results
5. **Review Feedback** - See detailed scores and personalized recommendations
6. **Download Results** - Save your analysis report as a text file

## 📊 Understanding Your Scores

### Overall Score Ranges
- **70-100**: Excellent - Professional quality speech
- **50-70**: Good - Solid performance with minor improvements possible
- **30-50**: Fair - Notable areas for improvement
- **0-30**: Needs Improvement - Significant practice recommended

### Analysis Dimensions

**🎵 Vocal Quality**
- Pitch variation and expressiveness
- Voice stability (jitter/shimmer)
- Harmonics-to-noise ratio

**🗣️ Articulation Clarity**
- Pronunciation precision
- Speech clarity
- Consonant emphasis

**🎭 Prosodic Variation**
- Intonation patterns
- Rhythm and stress
- Emotional expression

**⏱️ Speech Timing**
- Speech rate (syllables per second)
- Pause frequency and duration
- Overall pacing

## 💡 Tips for Best Results

- **Environment**: Record in a quiet space with minimal background noise
- **Microphone**: Use a good quality microphone for better analysis
- **Natural Speech**: Speak naturally as you would in a real interview
- **Preparation**: Think about your answer before recording
- **Practice**: Regular practice leads to improvement

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy from your repository
5. Set main file path: `interview_analyzer_app.py`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 📁 Project Structure

```
InterviewAnalysisApp/
├── interview_analyzer_app.py    # Main Streamlit application
├── speech_analyzer.py            # Speech feature extraction
├── confidence_scorer.py          # Scoring and feedback generation
├── interview_questions.json      # Interview question bank
├── config.json                   # Configuration and baselines
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── DEPLOYMENT.md                 # Deployment guide
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Audio analysis powered by [Librosa](https://librosa.org/) and [Parselmouth](https://parselmouth.readthedocs.io/)
- Inspired by the need for accessible interview practice tools

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ to help you ace your interviews!**

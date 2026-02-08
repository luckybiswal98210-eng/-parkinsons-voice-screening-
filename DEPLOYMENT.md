# 🚀 Deployment Guide

This guide covers deploying the Interview Analysis App locally and to Streamlit Cloud.

## 📋 Prerequisites

- Python 3.8+
- Git
- GitHub account (for Streamlit Cloud deployment)

## 🏠 Local Deployment

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/InterviewAnalysisApp.git
cd InterviewAnalysisApp

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run interview_analyzer_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 3. Troubleshooting Local Deployment

**Issue: Module not found errors**
```bash
# Ensure you're in the virtual environment
which python  # Should show path to venv/bin/python

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Issue: Audio recording not working**
- Check microphone permissions in your browser
- Ensure sounddevice can access your microphone
- Try using the upload option instead

**Issue: Streamlit port already in use**
```bash
# Run on a different port
streamlit run interview_analyzer_app.py --server.port 8502
```

## ☁️ Streamlit Cloud Deployment

### 1. Prepare Your Repository

Ensure your GitHub repository contains:
- ✅ `interview_analyzer_app.py`
- ✅ `speech_analyzer.py`
- ✅ `confidence_scorer.py`
- ✅ `interview_questions.json`
- ✅ `config.json`
- ✅ `requirements.txt`
- ✅ `.gitignore`

### 2. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/InterviewAnalysisApp`
   - Set branch: `main` (or `master`)
   - Set main file path: `interview_analyzer_app.py`

3. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete (2-5 minutes)
   - Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`

### 3. Streamlit Cloud Configuration

**Advanced Settings** (optional):
- Python version: 3.9 or higher
- Secrets: Not required for this app
- Resources: Default settings are sufficient

### 4. Troubleshooting Cloud Deployment

**Issue: Build fails with dependency errors**
- Check that `requirements.txt` is properly formatted
- Ensure all package names are correct
- Try pinning specific versions if needed:
  ```
  streamlit==1.28.0
  numpy==1.24.3
  librosa==0.10.1
  ```

**Issue: Audio recording doesn't work on deployed app**
- This is expected - browser security restrictions
- Users should use the "Upload audio file" option instead
- Consider adding a note in the app about this limitation

**Issue: App is slow or times out**
- Audio analysis can be CPU-intensive
- Consider adding progress indicators
- Optimize feature extraction if needed

## 🔧 Environment Variables

This app doesn't require environment variables, but you can add them if needed:

**In Streamlit Cloud:**
1. Go to App settings
2. Click "Secrets"
3. Add your secrets in TOML format:
   ```toml
   [general]
   app_name = "Interview Analysis"
   ```

**Locally:**
Create a `.streamlit/secrets.toml` file (already in .gitignore)

## 📊 Monitoring Your Deployment

### Streamlit Cloud
- View logs in the Streamlit Cloud dashboard
- Monitor app usage and performance
- Check for errors in real-time

### Local
- Terminal shows Streamlit logs
- Check for warnings or errors
- Monitor CPU/memory usage

## 🔄 Updating Your Deployment

### Streamlit Cloud
1. Push changes to your GitHub repository
2. Streamlit Cloud auto-deploys on push
3. Check deployment logs for success

### Local
```bash
# Pull latest changes
git pull origin main

# Restart Streamlit
# Press Ctrl+C to stop
streamlit run interview_analyzer_app.py
```

## 🔒 Security Considerations

- ✅ No sensitive data is stored
- ✅ Audio files are temporary and deleted after analysis
- ✅ No user authentication required
- ✅ No database connections
- ⚠️ Consider rate limiting for public deployments
- ⚠️ Monitor usage to prevent abuse

## 📈 Performance Optimization

**For better performance:**

1. **Reduce audio processing time**
   - Limit recording duration to 15-20 seconds
   - Use lower sample rates if acceptable

2. **Cache results**
   - Use Streamlit's `@st.cache_data` decorator
   - Cache configuration loading

3. **Optimize imports**
   - Import heavy libraries only when needed
   - Use lazy loading for analysis modules

## 🆘 Getting Help

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repository

## ✅ Deployment Checklist

Before deploying, ensure:

- [ ] All code is committed and pushed to GitHub
- [ ] `requirements.txt` is up to date
- [ ] `.gitignore` excludes unnecessary files
- [ ] README.md is complete and accurate
- [ ] App runs successfully locally
- [ ] All features work as expected
- [ ] No hardcoded sensitive information
- [ ] License file is included

---

**Happy Deploying! 🚀**

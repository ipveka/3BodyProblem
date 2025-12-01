# Deployment Guide

This guide explains how to deploy the 3BodyProblem N-Body Gravitational Simulation application to cloud hosting services like Render, Railway, or similar platforms.

## Prerequisites

Before deploying, ensure you have:
- A Git repository (GitHub, GitLab, or Bitbucket) with your code
- An account on your chosen deployment platform
- Python 3.11 or higher available on the deployment platform

## Deployment Files

The application uses two key files for deployment:

### `requirements.txt`
Contains all Python package dependencies needed to run the application:
- numpy, scipy (scientific computing)
- matplotlib, plotly, seaborn (visualization)
- pandas (data handling)
- streamlit (web framework)
- setuptools (build tools)

### `run_app.py`
The application launcher that:
- Checks Python version compatibility
- Verifies all dependencies are installed
- Validates project structure
- Launches Streamlit with proper configuration
- Handles environment variables (PORT, HOST)

## Deployment to Render

Render is a popular platform for deploying web applications. Follow these steps:

### Step 1: Prepare Your Repository

Ensure your code is pushed to a Git repository (GitHub, GitLab, or Bitbucket).

### Step 2: Create a New Web Service

1. Log in to [Render](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Select the repository containing your 3BodyProblem code

### Step 3: Configure Build Settings

In the Render dashboard, configure the following:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
python run_app.py
```

### Step 4: Configure Environment Variables (Optional)

Render will automatically set the `PORT` environment variable. The `run_app.py` script will use it automatically. If you need to customize:

- **PORT**: Usually set automatically by Render (default: 8501)
- **HOST**: Usually set automatically (default: 0.0.0.0)

You typically don't need to set these manually.

### Step 5: Deploy

1. Click "Create Web Service"
2. Render will:
   - Clone your repository
   - Install dependencies from `requirements.txt`
   - Run `python run_app.py`
   - Make your app available at a public URL

### Step 6: Monitor Deployment

- Check the build logs for any errors
- The app will be available at: `https://your-app-name.onrender.com`
- First deployment may take 5-10 minutes

## Deployment to Railway

Railway is another excellent option for Python applications:

### Step 1: Connect Repository

1. Log in to [Railway](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository

### Step 2: Configure Settings

In Railway dashboard:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
python run_app.py
```

### Step 3: Set Environment Variables

Railway automatically provides `PORT`. The `run_app.py` script handles this automatically.

### Step 4: Deploy

Railway will automatically deploy when you push to your repository.

## Deployment to Other Platforms

### Heroku

1. Create a `Procfile` in the root directory:
   ```
   web: python run_app.py
   ```

2. Deploy using Heroku CLI:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Streamlit Cloud (Easiest Option)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Set main file path: `app/app.py`
7. Deploy!

**Note**: Streamlit Cloud may require a `requirements.txt` file, which you already have.

## How `run_app.py` Works in Deployment

The `run_app.py` script is designed specifically for deployment environments:

1. **Pre-flight Checks**:
   - Verifies Python 3.11+ is available
   - Checks that all required packages are installed
   - Validates project structure

2. **Environment Detection**:
   - Reads `PORT` from environment (set by hosting platform)
   - Reads `HOST` from environment (defaults to 0.0.0.0)
   - Configures Streamlit accordingly

3. **Streamlit Launch**:
   - Runs Streamlit in headless mode (no browser)
   - Sets correct port and host for web access
   - Disables usage statistics collection

## Troubleshooting

### Build Fails: Missing Dependencies

**Problem**: Build fails with import errors.

**Solution**: 
- Verify `requirements.txt` includes all dependencies
- Check that package versions are compatible
- Review build logs for specific missing packages

### App Won't Start: Port Issues

**Problem**: Application fails to start or shows port errors.

**Solution**:
- Ensure `run_app.py` is using the `PORT` environment variable
- Check that your platform sets `PORT` automatically
- Verify the start command is `python run_app.py`

### Import Errors

**Problem**: ModuleNotFoundError or ImportError.

**Solution**:
- Ensure you're running from the project root
- Verify all files are in the repository
- Check that `core/` and `app/` directories exist

### Timeout Issues

**Problem**: Build or deployment times out.

**Solution**:
- Some platforms have timeout limits
- Ensure `requirements.txt` doesn't include unnecessary packages
- Consider using a `.dockerignore` if using Docker

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port number | 8501 | No (auto-set by platform) |
| `HOST` | Server address | 0.0.0.0 | No |

## Deployment Checklist

Before deploying, verify:

- [ ] `requirements.txt` is up to date with all dependencies
- [ ] `run_app.py` is executable and in the repository root
- [ ] All source files (`core/`, `app/`, `visualization/`) are committed
- [ ] Python 3.11+ is available on the deployment platform
- [ ] Build command: `pip install -r requirements.txt`
- [ ] Start command: `python run_app.py`
- [ ] Test locally: `python run_app.py` works on your machine

## Post-Deployment

After successful deployment:

1. **Test the Application**:
   - Visit your deployment URL
   - Try loading preset systems
   - Run a simulation
   - Verify visualizations work

2. **Monitor Performance**:
   - Check application logs
   - Monitor resource usage
   - Watch for any errors

3. **Update as Needed**:
   - Push changes to your repository
   - Platform will automatically redeploy (if configured)
   - Or manually trigger redeployment

## Cost Considerations

- **Render Free Tier**: Limited hours per month, spins down after inactivity
- **Railway Free Tier**: Limited usage, may require credit card
- **Streamlit Cloud**: Free tier available, very easy setup
- **Heroku**: No longer offers free tier

## Security Notes

- The application doesn't require authentication by default
- For production use, consider adding authentication
- Be mindful of resource limits on free tiers
- Monitor for abuse if making the app public

## Quick Reference

**Minimum Deployment Configuration:**
```
Build Command:  pip install -r requirements.txt
Start Command:   python run_app.py
Python Version:  3.11+
```

That's it! The `run_app.py` script handles everything else automatically, making deployment straightforward across different platforms.


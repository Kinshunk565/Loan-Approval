# 🚀 Deploying Loan Intelligence to Render

This guide provides step-by-step instructions to deploy your **Loan Approval Prediction System** to Render for free.

## 1. Prepare Your Repository
Ensure all your changes are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "Upgrade to Premium Vibrant Light UI"
git push origin main
```

## 2. Deploy Using Blueprint (Recommended)
Since the project already contains a `render.yaml` file, Render can automatically configure everything for you.

1.  Log in to [Render](https://dashboard.render.com).
2.  Click **"New +"** and select **"Blueprint"**.
3.  Connect your GitHub account and select this repository.
4.  Render will read `render.yaml` and show you the plan (Web Service + Build Command).
5.  Click **"Apply"**.

## 3. Manual Deployment (Alternative)
If you prefer to configure it manually:

1.  Click **"New +"** and select **"Web Service"**.
2.  Select your repository.
3.  **Runtime**: `Python`
4.  **Build Command**: 
    ```bash
    pip install -r requirements.txt && python -m src.model_training
    ```
5.  **Start Command**: 
    ```bash
    streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
    ```
6.  **Instance Type**: `Free`

### Advanced Settings:
-   **Environment Variables**:
    -   `PYTHON_VERSION`: `3.10.12`
    -   `STREAMLIT_SERVER_PORT`: `$PORT` (Render handles this automatically)

## 4. Verification
Once the build is complete (this may take 2-3 minutes):
1.  Click the URL provided (e.g., `https://loan-approval-predictor.onrender.com`).
2.  Ensure the "Live Prediction" and "EDA" pages load correctly.
3.  Test a prediction to verify the model was trained correctly during the build.

---
> [!TIP]
> Render's free tier services "spin down" after 15 minutes of inactivity. The first request after a long break might take a few seconds to start up.

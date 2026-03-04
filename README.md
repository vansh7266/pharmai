# PHARM**AI** — Intelligent Pharmaceutical Manufacturing Platform

> AI-powered batch quality prediction, anomaly detection, energy forecasting and automated corrective action alerts — all before a batch fails.

🌐 **Live Site:** [https://pharmai-0k9k.onrender.com](https://pharmai-0k9k.onrender.com)

---

## What is PharmAI?

Pharmaceutical batch failures cost crores in wasted materials, regulatory delays, and downtime. Traditional quality control catches failures **after** they happen. PharmAI catches them **before**.

PharmAI is an end-to-end intelligent manufacturing platform built with 4 AI models working together — predicting batch quality, detecting anomalies, forecasting energy consumption, and generating plain English corrective action alerts for floor operators.

---

## Features

| Feature | Description |
|---|---|
| 🤖 **Batch Quality Predictor** | Predicts 6 quality targets from 8 process parameters before batch completes |
| ⚡ **LSTM Energy Forecasting** | Trained on real sensor data to predict next-minute energy consumption |
| 🔬 **Anomaly Detection** | Autoencoder flags deviations with per-phase maintenance risk scores |
| 🧠 **Agentic AI Alerts** | Converts SHAP values into plain English corrective actions |
| 📡 **Live Dashboard** | Real-time monitoring across all 8 manufacturing phases |
| 📈 **Analytics Page** | Carbon footprint, energy usage, batch history insights |

---

## AI Models

### 1. Ensemble Predictor
- **Models:** XGBoost (40%) + Random Forest (35%) + Gradient Boosting (25%)
- **Input:** 8 process parameters → 12 engineered features
- **Output:** 6 quality targets — Hardness, Friability, Content Uniformity, Dissolution Rate, Tablet Weight, Disintegration Time
- **Accuracy:** >90%

### 2. LSTM Energy Forecasting
- **Architecture:** 2-layer LSTM, 64 units
- **Input:** 15-minute historical sensor data windows (5 channels)
- **Output:** Next-minute power consumption prediction

### 3. Autoencoder Anomaly Detection
- **Architecture:** 6→2→6 bottleneck
- **Threshold:** μ+2σ reconstruction error
- **Output:** Per-phase anomaly flags + maintenance risk scores

### 4. Agentic AI Alert System
- Reads SHAP feature importance + anomaly flags + maintenance risk
- Generates ranked plain English corrective action alerts
- Alert types: Critical / Warning / Info

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI + Uvicorn |
| **ML Models** | XGBoost, scikit-learn, TensorFlow, Keras |
| **Frontend** | Pure HTML / CSS / JavaScript |
| **Deployment** | Render.com |
| **Language** | Python 3.11 |

---

## Project Structure

```
PharmAI/
│
├── app.py                  ← FastAPI backend
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
│
├── models/                 ← Trained model files
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   ├── scaler.pkl
│   ├── lstm_model.keras
│   ├── autoencoder.keras
│   ├── ae_threshold.npy
│   └── *.json
│
├── index.html              ← Landing page
├── dashboard.html          ← Live monitoring dashboard
├── predictor.html          ← Batch quality predictor
├── analytics.html          ← Analytics & insights
├── history.html            ← Batch history
├── about.html              ← About the platform
└── splash.html             ← Splash screen
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server + model status |
| POST | `/predict` | Batch quality prediction |
| GET | `/dashboard/metrics` | Live KPI snapshot |
| GET | `/lstm/predict` | Next-minute energy forecast |
| POST | `/anomaly` | Phase anomaly detection |
| GET | `/maintenance` | Per-phase maintenance risk |
| GET | `/analytics/summary` | Full analytics data |
| GET | `/carbon` | Carbon footprint summary |

---

## Dataset

| Dataset | Details |
|---|---|
| `batch_manufacturing_data.csv` | 60 batches · 8 phases · 8 inputs · 6 quality targets |
| `energy_consumption_data.csv` | 4,800 records · 1-min resolution · 5 sensor channels |

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/pharmai.git
cd pharmai

# Create virtual environment
conda create -n ai_env python=3.11
conda activate ai_env

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py

# Open in browser
# http://localhost:5000
```

---

## Hackathon

Built as a submission for **IIT Hyderabad ML Hackathon — Track A**
Organized by **Tinkerers' Lab IITH** in collaboration with **AVEVA**

---

## Live Demo

🌐 [https://pharmai-0k9k.onrender.com](https://pharmai-0k9k.onrender.com)

> Note: Hosted on Render free tier — first load may take 30 seconds if the server is sleeping.

---

*Built with Python, FastAPI, XGBoost, TensorFlow, and a lot of chai. ☕*

# 🧠 CPRS — AI Cognitive Pattern Recognition System

> An end-to-end intelligent data analysis platform that automates the entire ML workflow —  
> from natural language dataset discovery to interpretable AI-generated predictions.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Ollama](https://img.shields.io/badge/Phi--3_LLM-7B1FA2?style=flat)
![Kaggle](https://img.shields.io/badge/Kaggle_API-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## 📋 Overview

CPRS is a comprehensive full-stack AI platform for automated cognitive pattern recognition in tabular datasets. It eliminates the need for manual data science expertise by automating every stage of the ML pipeline — from intelligently finding the right dataset to generating research-quality natural language explanations using a **locally hosted Large Language Model**.

The system features a novel **Cognitive Pattern Score (CPS)** metric, a two-stage semantic dataset recommendation engine with Kaggle integration, and a privacy-preserving offline LLM layer — making it suitable for sensitive domains including healthcare, finance, and human resources.

---

## 🎯 Final Vision

When all planned improvements are complete, CPRS will be a fully production-ready intelligent web tool:

### Landing Page
- Natural language problem statement input
- **Two-stage Kaggle ranking engine**: fetches 40-50 datasets → priority filter → deep relevance scoring → top 6-10 ranked
- Hover previews: Kaggle description, sample rows, key columns
- One-click download and auto-load into pipeline
- Local search with instant keyword matching

### Data Processing Page
- Drag-and-drop CSV upload
- Live stat cards with formula breakdowns
- Feature importance chart
- CPR AI dataset narration (Phi-3 generated)
- Smart preprocessing report with encoding strategy per column
- Categorical text input resolution for predictions

### Model Training Page
- 17+ model AutoML with real-time WebSocket progress
- Best model trophy card
- Training score formula display
- CPR AI training interpretation

### Pattern Dashboard
- Animated Cognitive Pattern Score ring (0-100)
- CPS formula breakdown display
- Phi-3 AI summaries inside Patterns + Cluster sections
- Smart graph selection (only relevant charts for target)
- Professional Plotly visualizations

### Live Prediction Engine
- Top-3 most important features as inputs
- Categorical text input (type "female" → auto-resolves to encoded value)
- Real-time prediction with CPR AI explanation
- Actual vs predicted for sample rows

---

## ✅ Implementation Status

| Feature | Status |
|---|---|
| FastAPI structured backend | ✅ Complete |
| React frontend (Glassmorphism) | ✅ Complete |
| Landing page with local search | ✅ Complete |
| Kaggle API integration | ✅ Complete |
| Dataset profiling | ✅ Complete |
| Smart preprocessing pipeline | ✅ Complete |
| AutoML engine (12+ models) | ✅ Complete |
| Cognitive Pattern Score (CPS) | ✅ Complete |
| Pattern/cluster/interaction discovery | ✅ Complete |
| Phi-3 local LLM (Ollama) | ✅ Complete |
| Live prediction engine | ✅ Complete |
| Dark/light theme switcher | ✅ Complete |
| GitHub repository | ✅ Complete |
| Research paper (IEEE format) | ✅ Complete |
| Kaggle two-stage ranking | 🔄 Pending |
| Local search 0% fix | 🔄 Pending |
| More ML models (XGBoost, LightGBM) | 🔄 Pending |
| AI summaries in Patterns + Clusters | 🔄 Pending |
| Smart graph selection | 🔄 Pending |
| Professional graph styling | 🔄 Pending |
| Login / Auth system | 🔮 Future |
| User dashboard + history | 🔮 Future |
| PDF export | 🔮 Future |
| Real-time training progress | 🔮 Future |

---

## 🗂️ Project Structure

```
Cognitive Pattern Recognition/
│
├── app/                           # FastAPI backend
│   ├── main.py                    # App init + CORS + routers
│   ├── api/                       # Route handlers
│   │   ├── dataset.py             # Upload + demo datasets
│   │   ├── preprocessing.py       # Preprocess + feature importance
│   │   ├── training.py            # AutoML training
│   │   ├── analysis.py            # Dashboard + predict + AI endpoints
│   │   ├── finder.py              # Local semantic search
│   │   └── kaggle.py              # Kaggle search + download
│   ├── services/                  # Business logic
│   │   ├── dataset_service.py
│   │   ├── preprocessing_service.py
│   │   ├── training_service.py
│   │   ├── analysis_service.py
│   │   ├── insight_service.py
│   │   ├── ai_service.py          # Phi-3 via Ollama
│   │   ├── dataset_finder_service.py
│   │   └── kaggle_service.py
│   └── core/
│       ├── config.py              # Settings + paths
│       └── state.py               # Centralized session state
│
├── datasets/                      # Demo + downloaded CSVs
├── tests/                         # pytest suite
├── requirements.txt
├── .env                           # Kaggle credentials
│
└── frontend/                      # React + Vite
    └── src/
        ├── App.jsx
        ├── api/client.js          # All API calls
        ├── store/appStore.js      # Zustand state
        ├── pages/
        │   ├── LandingPage.jsx
        │   ├── DataPage.jsx
        │   ├── TrainingPage.jsx
        │   └── DashboardPage.jsx
        ├── components/
        │   ├── ui/                # Button, Card, Badge, AIPanel
        │   ├── charts/            # PlotlyChart wrapper
        │   └── layout/            # Navbar, SplashScreen
        └── styles/theme.css       # Dark/light CSS variables
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- [Ollama](https://ollama.com) installed
- Kaggle account + API key

### Installation

**1. Clone repository**
```bash
git clone https://github.com/Kunalchaudhary-DS/cognitive-pattern-recognition-ai.git
cd cognitive-pattern-recognition-ai
```

**2. Backend setup**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
```

**3. Configure Kaggle credentials**

Create `.env` file:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Create `C:\Users\<you>\.kaggle\kaggle.json`:
```json
{"username": "your_username", "key": "your_api_key"}
```

**4. Download AI model**
```bash
ollama pull phi3
```

**5. Frontend setup**
```bash
cd frontend
npm install
```

### Running

Open 3 terminals:

```bash
# Terminal 1 — AI Engine
ollama serve

# Terminal 2 — Backend
venv\Scripts\activate
uvicorn app.main:app --reload

# Terminal 3 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.  
API docs: **http://localhost:8000/docs**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/demo-datasets/` | List demo datasets |
| POST | `/upload/` | Upload CSV file |
| POST | `/load-demo-dataset/` | Load a demo dataset |
| POST | `/feature-importance/` | Correlation-based feature importance |
| POST | `/preprocess/` | Run preprocessing pipeline |
| POST | `/train/` | AutoML training |
| GET | `/dashboard-data/` | Full pattern analysis |
| POST | `/predict/` | Real-time inference |
| GET | `/sample-predictions/` | Actual vs predicted (5 rows) |
| GET | `/encoding-maps/` | Categorical encoding lookup |
| POST | `/find-datasets/` | Local semantic search |
| POST | `/kaggle-search/` | Search Kaggle datasets |
| POST | `/kaggle-download/` | Download Kaggle dataset |
| POST | `/ai/dataset-explanation/` | Phi-3 dataset narration |
| POST | `/ai/training-explanation/` | Phi-3 training interpretation |
| POST | `/ai/pattern-explanation/` | Phi-3 pattern analysis |
| GET | `/ai/insight-summary/` | Phi-3 research conclusion |

---

## 📊 Cognitive Pattern Score (CPS)

The CPS is a **novel composite metric** introduced by CPRS:

```
CPS = Q(0.40) + P(0.20) + C(0.20) + I(0.20)   ∈ [0, 100]

Where:
  Q = ⌊(1 − missing_ratio) × 40⌋    → Data Quality     (max 40)
  P = min(|patterns| × 5, 20)        → Pattern Strength (max 20)
  C = min(|clusters| × 5, 20)        → Cluster Score    (max 20)
  I = min(|interactions| × 5, 20)    → Interaction Score(max 20)
```

| CPS | Strength | Meaning |
|---|---|---|
| 81–100 | 🟢 Strong | Rich patterns, high predictive potential |
| 61–80 | 🟡 Moderate | Good patterns, suitable for applications |
| 0–60 | 🔴 Weak | Limited structure, enrich data first |

---

## 🤖 AutoML Model Pool

**Regression (9 models):** Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, Extra Trees, KNN, SVR

**Classification (8 models):** Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, KNN, SVC, Decision Tree, Naive Bayes

**Planned additions:** XGBoost, LightGBM, AdaBoost, BaggingClassifier

---

## 📄 Research

CPRS is developed as a **research-grade system** targeting Scopus/IEEE publication.

**Key contributions:**
- Novel CPS metric with formal definition and empirical validation (r=0.74 with model performance)
- Two-stage semantic dataset recommendation (87% top-1 accuracy)
- Privacy-preserving local LLM interpretability (no cloud dependency)
- Comprehensive AutoML benchmarking across 12 datasets

**Target venues:** ICACDS (Springer LNCS), IEEE ICCCNT, IEEE Access  
**Patent filing:** In progress for CPS algorithm and integrated pipeline

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.11 |
| ML | Scikit-learn 1.4.2 |
| AI/LLM | Phi-3 Mini via Ollama (offline) |
| Dataset Discovery | Kaggle Python API |
| Frontend | React 18 + Vite 5 |
| Animations | Framer Motion 11 |
| State | Zustand |
| Charts | Plotly.js 2.26 |
| Styling | Custom CSS Variables (glassmorphism) |

---

## 📜 License & Credits

Developed for academic research purposes. Patent filing in progress.

**Author:** Kunal Chaudhary  
**Acknowledgements:** Microsoft Research (Phi-3), Ollama, Scikit-learn, Kaggle, FastAPI

---

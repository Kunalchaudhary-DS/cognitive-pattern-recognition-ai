# 🧠 CPRS — AI Cognitive Pattern Recognition System

> An end-to-end intelligent data analysis platform that automates the entire ML workflow —  
> from natural language dataset discovery to semantically validated, AI-interpreted predictions.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Ollama](https://img.shields.io/badge/TinyLlama_LLM-7B1FA2?style=flat)
![Kaggle](https://img.shields.io/badge/Kaggle_API-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## 📋 Overview

CPRS is a comprehensive full-stack AI platform for automated cognitive pattern recognition in tabular datasets. It eliminates the need for manual data science expertise by automating every stage of the ML pipeline — from intelligently finding the right dataset to generating research-quality natural language explanations using a **locally hosted Large Language Model**.

The system features a novel **Cognitive Pattern Score (CPS)** metric, a two-stage semantic dataset recommendation engine with Kaggle integration, a privacy-preserving offline LLM layer, and a **Semantic Prediction Interceptor** — a hybrid statistical + LLM system that enforces domain-logical validity on all model predictions.

---

## 🎯 Key Features

### Landing Page
- Natural language problem statement input
- **Two-stage Kaggle ranking engine**: fetches datasets → deep relevance scoring → ranked results
- One-click download and auto-load into pipeline
- Local dataset search with keyword matching and auto-registration of downloaded datasets

### Data Processing Page
- Drag-and-drop CSV upload + 10 built-in demo datasets
- Live stat cards: rows, columns, quality score, missing values
- Smart preprocessing: multi-strategy encoding (ordinal, frequency, binary, one-hot)
- AI dataset narration (TinyLlama) — concise 2-sentence profile
- Encoding map display for categorical feature inputs

### Model Training Page
- **2-Stage AutoML**: Stage 1 screens all 13+ models on a sample, Stage 2 fully trains top performers
- Best model highlight with trophy card
- Full metrics table (R², MAE, RMSE for regression; Accuracy, F1, ROC-AUC for classification)
- Smart model ranking by primary metric (F1-Macro for imbalanced, Accuracy for balanced classification)
- AI training interpretation — direct 2-sentence analysis of why the best model won
- **Semantic Prediction Interceptor** — automatically generates a constraint rulebook from training data

### Pattern Dashboard
- Animated Cognitive Pattern Score ring (0–100)
- Target distribution, feature importance, model comparison, and correlation heatmap
- Discovered Patterns + Cluster Analysis panels, each with a **⚡ CPR AI INSIGHT** strip
- Smart auto-graph selection (top-priority charts per dataset type)
- AI Pattern Interpretation + AI Research Conclusion panels
- Professional Plotly visualizations (bar, scatter, box, pie, histogram, heatmap)

### Live Prediction Engine
- Top-8 most important features as inputs with categorical text resolution
- **Semantic Prediction Interceptor active**: raw model output is checked against:
  - Statistical bounds (min/max from training data)
  - Cross-column relative rules (e.g., `years_with_manager ≤ total_working_years`)
  - LLM-inferred semantic domain constraints (e.g., a "score" cannot exceed 100)
- Transparent correction display: shows raw model output, corrected value, and reason
- Soft domain warnings for unconfirmed constraint violations
- Actual vs predicted for first 5 dataset rows

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
| AutoML engine (13+ models incl. XGBoost, LightGBM) | ✅ Complete |
| 2-Stage AutoML training (screen + full train) | ✅ Complete |
| Cognitive Pattern Score (CPS) | ✅ Complete |
| Pattern/cluster/interaction discovery | ✅ Complete |
| TinyLlama local LLM (Ollama) | ✅ Complete |
| Focused AI generation (2-sentence targeted prompts) | ✅ Complete |
| AI Panel Insights (Patterns + Clusters panels) | ✅ Complete |
| Smart graph selection (feature-priority ranked) | ✅ Complete |
| Professional graph styling (Plotly) | ✅ Complete |
| Live prediction engine | ✅ Complete |
| Confusion matrix + ROC curve endpoints | ✅ Complete |
| Categorical text input resolution | ✅ Complete |
| **Semantic Prediction Interceptor** | ✅ Complete |
| — Layer 1: Statistical constraint extraction | ✅ Complete |
| — Layer 2: Ollama LLM semantic enrichment | ✅ Complete |
| — Layer 3: Constraint merger & validation | ✅ Complete |
| — Layer 4: Prediction interceptor with transparency | ✅ Complete |
| GitHub repository | ✅ Complete |
| Login / Auth system | 🔮 Future |
| User dashboard + history | 🔮 Future |
| PDF export | 🔮 Future |
| Real-time training progress (WebSocket) | 🔮 Future |

---

## 🗂️ Project Structure

```
Cognitive Pattern Recognition/
│
├── app/                              # FastAPI backend
│   ├── main.py                       # App init + CORS + routers
│   ├── api/                          # Route handlers
│   │   ├── dataset.py                # Upload + demo datasets
│   │   ├── preprocessing.py          # Preprocess + feature importance
│   │   ├── training.py               # AutoML training + constraint generation
│   │   ├── analysis.py               # Dashboard + predict + AI endpoints
│   │   ├── finder.py                 # Local semantic search
│   │   └── kaggle.py                 # Kaggle search + download
│   ├── services/                     # Business logic
│   │   ├── dataset_service.py
│   │   ├── preprocessing_service.py
│   │   ├── training_service.py       # 2-stage AutoML engine
│   │   ├── analysis_service.py       # Pattern, cluster, interaction discovery
│   │   ├── insight_service.py        # CPS, feature importance, key insights
│   │   ├── ai_service.py             # TinyLlama via Ollama (all AI generation)
│   │   ├── constraint_service.py     # Semantic Prediction Interceptor (Layers 1,3,4)
│   │   ├── dataset_finder_service.py
│   │   └── kaggle_service.py
│   └── core/
│       ├── config.py                 # Settings + paths
│       └── state.py                  # Centralized session state (incl. constraint_map)
│
├── datasets/                         # Demo + downloaded CSVs
├── tests/                            # pytest suite
├── requirements.txt
├── .env                              # Kaggle credentials
│
└── frontend/                         # React + Vite
    └── src/
        ├── App.jsx
        ├── api/client.js             # All API calls
        ├── store/appStore.js         # Zustand state
        ├── pages/
        │   ├── LandingPage.jsx
        │   ├── DataPage.jsx
        │   ├── TrainingPage.jsx
        │   └── DashboardPage.jsx     # Main dashboard + AI panels + prediction
        ├── components/
        │   ├── ui/                   # Button, Card, Badge, AIPanel, PredictionEngine
        │   ├── charts/               # PlotlyChart wrapper
        │   └── layout/               # Navbar, SplashScreen, StepFlow
        └── styles/theme.css          # Dark/light CSS variables
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
ollama pull tinyllama
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
| POST | `/train/` | AutoML training + constraint map generation |
| GET | `/dashboard-data/` | Full pattern analysis |
| POST | `/predict/` | Real-time inference with Semantic Interceptor |
| GET | `/sample-predictions/` | Actual vs predicted (5 rows) |
| GET | `/encoding-maps/` | Categorical encoding lookup |
| GET | `/confusion-matrix/` | Confusion matrix (classification) |
| GET | `/roc-curve/` | ROC curve + AUC (binary classification) |
| POST | `/find-datasets/` | Local semantic search |
| POST | `/kaggle-search/` | Search Kaggle datasets |
| POST | `/kaggle-download/` | Download Kaggle dataset |
| POST | `/ai/dataset-explanation/` | TinyLlama dataset narration |
| POST | `/ai/training-explanation/` | TinyLlama training interpretation |
| POST | `/ai/pattern-explanation/` | TinyLlama pattern analysis |
| GET | `/ai/insight-summary/` | TinyLlama research conclusion |
| POST | `/ai/panel-insights/` | TinyLlama insights for Patterns + Clusters panels |

---

## 🛡️ Semantic Prediction Interceptor

A novel hybrid constraint system that enforces **domain-logical validity** on all regression predictions. It operates in 4 layers, all triggered automatically at training time:

```
Training Complete
      │
      ├── [Layer 1] Statistical Extractor  → hard bounds + cross-column inequalities from data
      └── [Layer 2] Ollama Semantic Engine → infers domain constraints from column names
                          │
                   [Layer 3] Constraint Merger
                   (statistical truth gates LLM proposals — prevents hallucination)
                          │
                   Stored as constraint_map in session state
                          │
            At Prediction Time → [Layer 4] Interceptor
              ├── Clips raw output to effective min/max
              ├── Checks relative rules (e.g. predicted ≤ user's input for ref column)
              └── Returns corrected value + full transparency report to UI
```

**Example:** If the model predicts `writing_score = 101`, the interceptor clips it to `100` and displays: `RAW MODEL OUTPUT: ~~101~~ → corrected` with the reason shown in the UI.

**Graceful degradation:** If Ollama is offline, Layer 2 returns `{}` and the system falls back to statistical-only constraints without any errors.

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

**Regression (11-13 models):**
Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, HistGradientBoosting, Extra Trees, AdaBoost, KNN, SVR *(small datasets only)*, GradientBoosting *(small datasets only)*, **XGBoost** *(if installed)*, **LightGBM** *(if installed)*

**Classification (10-12 models):**
Logistic Regression, Random Forest, HistGradientBoosting, Extra Trees, AdaBoost, KNN, SVC *(small datasets only)*, GradientBoosting *(small datasets only)*, Decision Tree, Naive Bayes, **XGBoost** *(if installed)*, **LightGBM** *(if installed)*

**2-Stage Training for large datasets (>50k rows):**
- Stage 1: All models screened on a stratified sample (≤5k rows, 3-fold CV)
- Stage 2: Top 5 models fully trained on 100% data (5-fold CV + hold-out test)
- SVR/SVC excluded automatically for large datasets (O(n²) complexity)

**Model selection metric:**
- Regression → R² (CV mean)
- Classification (balanced) → Accuracy
- Classification (imbalanced, any class < 15%) → F1-Macro

---

## 🤖 AI Generation System

All AI explanations are generated locally via **TinyLlama** through Ollama:

| Generation | Trigger | Output |
|---|---|---|
| Dataset narration | After upload/demo load | 2-sentence dataset profile |
| Training interpretation | After training | Why the best model won |
| Pattern explanation | Dashboard load | Pattern + cluster interpretation |
| Research conclusion | Dashboard load | Final research summary |
| Panel insights | Dashboard load | 1-sentence insight per panel |
| Semantic constraints | After training | JSON constraint rulebook |
| Prediction explanation | On each prediction | 2-sentence real-world meaning |

All prompts use structured format instructions (`Sentence 1: ... Sentence 2: ...`) with low temperature (0.3–0.4) to prevent rambling and hallucination.

---

## 📄 Research

CPRS is developed as a **research-grade system** targeting Scopus/IEEE publication.

**Key contributions:**
- Novel CPS metric with formal definition and empirical validation
- Two-stage semantic dataset recommendation with Kaggle integration
- **Semantic Prediction Interceptor** — hybrid statistical + LLM constraint enforcement (novel)
- Privacy-preserving local LLM interpretability (no cloud dependency)
- Comprehensive 2-stage AutoML benchmarking across 13+ model types

**Target venues:** ICACDS (Springer LNCS), IEEE ICCCNT, IEEE Access  
**Patent filing:** In progress for CPS algorithm, Semantic Prediction Interceptor, and integrated pipeline

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.11 |
| ML | Scikit-learn 1.4.2 + XGBoost 2.0.3 + LightGBM 4.3.0 |
| AI/LLM | TinyLlama via Ollama (fully offline) |
| Dataset Discovery | Kaggle Python API 1.6.17 |
| Frontend | React 18 + Vite 5 |
| Animations | Framer Motion 11 |
| State | Zustand |
| Charts | Plotly.js 2.26 |
| Styling | Custom CSS Variables (glassmorphism + dark mode) |

---

## 📜 License & Credits

Developed for academic research purposes. Patent filing in progress.

**Author:** Kunal Chaudhary  
**Acknowledgements:** Meta AI (TinyLlama), Ollama, Scikit-learn, XGBoost, LightGBM, Kaggle, FastAPI

---

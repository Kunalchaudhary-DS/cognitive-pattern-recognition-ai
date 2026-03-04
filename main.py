from fastapi import FastAPI, UploadFile, File, Request , Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder


# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Linear Models
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet
)

# Tree & Ensemble Models
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier
)

# Neighbors
from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier
)

# Support Vector Machines
from sklearn.svm import SVR, SVC

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Model Selection
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)

import pandas as pd
import numpy as np
import io
import os

app = FastAPI(title="Cognitive Pattern Recognition System")

templates = Jinja2Templates(directory="templates")
stored_df = None
stored_X = None
stored_y = None
stored_problem_type = None
stored_training_results = None
stored_target_column = None
stored_best_model = None
stored_feature_names = None
stored_scaler = None
stored_needs_scaling = False
stored_problem_type = None
stored_preprocessor = None
stored_strong_correlations = []
DATASET_FOLDER = "datasets"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/demo-datasets/")
async def get_demo_datasets():

    dataset_info = [
        {
            "file": "accounting_dataset.csv",
            "name": "Accounting Dataset",
            "category": "Finance"
        },
        {
            "file": "cricket_mini.csv",
            "name": "Cricket Statistics Dataset",
            "category": "Sports"
        },
        {
            "file": "Global_Education.csv",
            "name": "Global Education Dataset",
            "category": "Education"
        },
        {
            "file": "healthcare_dataset.csv",
            "name": "Healthcare Dataset",
            "category": "Healthcare"
        },
        {
            "file": "healthcare-dataset-stroke-data.csv",
            "name": "Stroke Prediction Dataset",
            "category": "Healthcare"
        },
        {
            "file": "HR_Attrition.csv",
            "name": "HR Attrition Dataset",
            "category": "Human Resources"
        },
        {
            "file": "International_Education_Costs.csv",
            "name": "International Education Costs Dataset",
            "category": "Education"
        },
        {
            "file": "personal_finance_tracker_dataset.csv",
            "name": "Personal Finance Tracker Dataset",
            "category": "Finance"
        },
        {
            "file": "retail-grocery-customers.csv",
            "name": "Retail Grocery Customers Dataset",
            "category": "Retail"
        },
        {
            "file": "Smartphone_Usage_Productivity_Dataset_50000.csv",
            "name": "Smartphone Usage & Productivity Dataset",
            "category": "Technology"
        },
        {
            "file": "StudentsPerformance.csv",
            "name": "Students Performance Dataset",
            "category": "Education"
        },
        {
            "file": "synthetic_personal_finance_dataset.csv",
            "name": "Synthetic Personal Finance Dataset",
            "category": "Finance"
        }
    ]

    return {"datasets": dataset_info}


@app.post("/load-demo-dataset/")
async def load_demo_dataset(dataset_name: str = Form(...)):

    global stored_df

    file_path = os.path.join(DATASET_FOLDER, dataset_name)

    if not os.path.exists(file_path):
        return {"error": "Dataset not found"}

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")
    stored_df = df.copy()

    # ======================================================
    # DATASET VALIDATION
    # ======================================================

    # Check minimum rows
    if df.shape[0] < 5:
        return {"error": "Dataset must contain at least 5 rows"}

    # Check minimum columns
    if df.shape[1] < 2:
        return {"error": "Dataset must contain at least 2 columns"}

    # Check if dataset is empty
    if df.empty:
        return {"error": "Uploaded dataset is empty"}

    # Check if all columns are empty
    if df.dropna(how="all").shape[0] == 0:
        return {"error": "Dataset contains only missing values"}

    # preview
    preview_df = df.head().copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    full_df = df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.astype(object).where(pd.notnull(full_df), None)

    # column types
    numerical_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # missing values
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df) * 100).round(2)

    
    # ======================================================
    # DATASET PROFILE SUMMARY
    # ======================================================

    total_missing = missing_counts.sum()
    total_cells = df.shape[0] * df.shape[1]

    missing_ratio = total_missing / total_cells

    missing_percent_dataset = round(missing_ratio * 100, 2)


    # Detect problem type suggestion
    unique_counts = df.nunique()

    classification_candidates = [
        col for col in df.columns
        if unique_counts[col] <= 10 and col in categorical_cols
    ]

    if classification_candidates:
        suggested_problem = "Classification"
    else:
        suggested_problem = "Regression"


    # Data quality score
    quality_score = round(100 - missing_percent_dataset, 2)
    

    missing_summary = {
        col: {
            "count": int(missing_counts[col]),
            "percentage": float(missing_percentage[col])
        }
        for col in df.columns if missing_counts[col] > 0
    }

    # correlations
    strong_correlations = []

    if len(numerical_cols) > 1:

        correlation_matrix = df[numerical_cols].corr()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):

                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                value = correlation_matrix.iloc[i, j]

                if abs(value) > 0.7:
                    strong_correlations.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(value), 2)
                    })

    return {
        "rows": int(len(df)),
        "total_columns": len(df.columns),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "strong_correlations": strong_correlations,
        "columns": list(df.columns),
        "profile_summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_percent": missing_percent_dataset,
            "suggested_problem": suggested_problem,
            "quality_score": quality_score
        },
        "preview": preview_df.to_dict(orient="records"),
        "full_data": full_df.to_dict(orient="records")

    }


@app.post("/upload/")
async def upload_dataset(file: UploadFile = File(...)):

    global stored_df

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    stored_df = df.copy()

    # ======================================================
    # DATASET VALIDATION
    # ======================================================

    # Check minimum rows
    if df.shape[0] < 5:
        return {"error": "Dataset must contain at least 5 rows"}

    # Check minimum columns
    if df.shape[1] < 2:
        return {"error": "Dataset must contain at least 2 columns"}

    # Check if dataset is empty
    if df.empty:
        return {"error": "Uploaded dataset is empty"}

    # Check if all columns are empty
    if df.dropna(how="all").shape[0] == 0:
        return {"error": "Dataset contains only missing values"}

    # ======================================================
    # PREVIEW CLEANING
    # ======================================================

    preview_df = df.head().copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    full_df = df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.astype(object).where(pd.notnull(full_df), None)

    # ======================================================
    # BASIC STRUCTURE
    # ======================================================

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    total_rows = len(df)
    total_columns = len(df.columns)

    # ======================================================
    # CORRELATION (Avoid duplicate pairs)
    # ======================================================

    strong_correlations = []

    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):

                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                value = correlation_matrix.iloc[i, j]

                if abs(value) > 0.7:
                    strong_correlations.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(value), 2)
                    })
    global stored_strong_correlations
    stored_strong_correlations = strong_correlations

    # ======================================================
    # MISSING VALUES
    # ======================================================

    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / total_rows * 100).round(2)

    # ======================================================
    # DATASET PROFILE SUMMARY
    # ======================================================

    total_missing = missing_counts.sum()
    total_cells = df.shape[0] * df.shape[1]

    missing_ratio = total_missing / total_cells

    missing_percent_dataset = round(missing_ratio * 100, 2)


    # Detect problem type suggestion
    unique_counts = df.nunique()

    classification_candidates = [
        col for col in df.columns
        if unique_counts[col] <= 10 and col in categorical_cols
    ]

    if classification_candidates:
        suggested_problem = "Classification"
    else:
        suggested_problem = "Regression"


    # Data quality score
    quality_score = round(100 - missing_percent_dataset, 2)

    missing_summary = {
        col: {
            "count": int(missing_counts[col]),
            "percentage": float(missing_percentage[col])
        }
        for col in df.columns if missing_counts[col] > 0
    }

    total_missing = df.isnull().sum().sum()

    # ======================================================
    # DUPLICATES
    # ======================================================

    duplicate_count = int(df.duplicated().sum())

    # ======================================================
    # DATASET NATURE
    # ======================================================

    if len(numerical_cols) > len(categorical_cols):
        dataset_nature = "Mostly Numerical Dataset"
    elif len(categorical_cols) > len(numerical_cols):
        dataset_nature = "Mostly Categorical Dataset"
    else:
        dataset_nature = "Balanced Dataset"

    # ======================================================
    # CLASS IMBALANCE CHECK
    # ======================================================

    class_imbalance = None

    for col in categorical_cols:
        if df[col].nunique() <= 10:
            distribution = df[col].value_counts(normalize=True)
            if distribution.max() > 0.8:
                class_imbalance = f"Column '{col}' shows strong class imbalance."
                break

    # ======================================================
    # DATA QUALITY SCORE
    # ======================================================

    missing_ratio = total_missing / (total_rows * total_columns)

    quality_score = 100

    if missing_ratio > 0.1:
        quality_score -= 20

    if duplicate_count > 0:
        quality_score -= 10

    if class_imbalance:
        quality_score -= 10

    quality_score = max(0, quality_score)

    # ======================================================
    # DATASET SUMMARY (SMART PARAGRAPH)
    # ======================================================

    dataset_summary = (
        f"The dataset contains {total_rows} rows and {total_columns} columns. "
        f"It includes {len(numerical_cols)} numerical features and "
        f"{len(categorical_cols)} categorical features. "
        f"{dataset_nature}. "
    )

    if total_missing > 0:
        dataset_summary += "Missing values are present. "
    else:
        dataset_summary += "No missing values detected. "

    if duplicate_count > 0:
        dataset_summary += f"There are {duplicate_count} duplicate rows. "

    if class_imbalance:
        dataset_summary += class_imbalance

    # ======================================================
    # BASIC STATS
    # ======================================================

    basic_stats = df.describe(include="all").fillna("").to_dict()

    # ======================================================
    # RESPONSE
    # ======================================================

    response_data = {
        "rows": total_rows,
        "total_columns": total_columns,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "duplicate_count": duplicate_count,
        "quality_score": quality_score,
        "dataset_nature": dataset_nature,
        "class_imbalance": class_imbalance,
        "dataset_summary": dataset_summary,
        "basic_statistics": basic_stats,
        "strong_correlations": strong_correlations,
        "profile_summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_percent": missing_percent_dataset,
            "suggested_problem": suggested_problem,
            "quality_score": quality_score
        },
        "columns": list(df.columns),
        "preview": preview_df.to_dict(orient="records"),
        "full_data": full_df.to_dict(orient="records")
    }

    return JSONResponse(content=response_data)

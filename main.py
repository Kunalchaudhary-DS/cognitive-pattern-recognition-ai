from fastapi import FastAPI, UploadFile, File, Request , Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score ,StratifiedKFold, KFold
import pandas as pd
import numpy as np
import io

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
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_dataset(file: UploadFile = File(...)):
    global stored_df

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    stored_df = df.copy()

    # Preview
    preview_df = df.head().copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    # Structural Analysis
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    correlation_matrix = df[numerical_cols].corr()

    strong_correlations = []

    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2:
                value = correlation_matrix.loc[col1, col2]
                if abs(value) > 0.7:
                    strong_correlations.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(value), 2)
                    })

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df) * 100).round(2)

    missing_summary = {
        col: {
            "count": int(missing_counts[col]),
            "percentage": float(missing_percentage[col])
        }
        for col in df.columns if missing_counts[col] > 0
    }

    basic_stats = df.describe().to_dict()

    response_data = {
        "rows": int(len(df)),
        "total_columns": len(df.columns),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "basic_statistics": basic_stats,
        "strong_correlations": strong_correlations,
        "columns": list(df.columns),
        "preview": preview_df.to_dict(orient="records")
    }

    return JSONResponse(content=response_data)

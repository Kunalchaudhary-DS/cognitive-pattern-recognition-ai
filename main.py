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

@app.post("/feature-importance/")
async def feature_importance(target_column: str = Form(...)):
    global stored_df

    if stored_df is None:
        return JSONResponse(content={"error": "No dataset uploaded"})

    df = stored_df.copy()

    if target_column not in df.columns:
        return JSONResponse(content={"error": "Invalid target column"})

    # Detect problem type
    target_dtype = df[target_column].dtype
    unique_values = df[target_column].nunique()

    if target_dtype == "object" or unique_values <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"

    # For classification → encode target temporarily
    if problem_type == "classification":
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    # Select only numerical features
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_column not in numerical_cols:
        return JSONResponse(content={
            "error": "Target encoding failed"
        })

    correlations = df[numerical_cols].corr()[target_column].drop(target_column)
    ranked = correlations.abs().sort_values(ascending=False)

    result = [
        {
            "feature": feature,
            "correlation": round(float(correlations[feature]), 3)
        }
        for feature in ranked.index
    ]

    return JSONResponse(content={
        "feature_importance": result,
        "problem_type": problem_type
    })

@app.post("/preprocess/")
async def preprocess_data(target_column: str = Form(...)):
    global stored_df, stored_X, stored_y, stored_problem_type, stored_target_column 
    global stored_preprocessor, stored_feature_names

    if stored_df is None:
        return JSONResponse(content={"error": "No dataset uploaded"})

    df = stored_df.copy()

    if target_column not in df.columns:
        return JSONResponse(content={"error": "Invalid target column"})

    
    # Detect Problem Type
    target_dtype = df[target_column].dtype
    unique_values = df[target_column].nunique()

    if target_dtype == "object" or unique_values <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"

    original_shape = df.shape

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])

    dropped_rows = original_shape[0] - df.shape[0]

    # Impute Missing Values 
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if numerical_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Separate Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Smart Encoding Strategy
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    binary_columns = []
    low_cardinality = []
    high_cardinality = []

    for col in cat_features:
        unique_count = X[col].nunique()

        if unique_count == 2:
            binary_columns.append(col)
        elif unique_count <= 10:
            low_cardinality.append(col)
        else:
            high_cardinality.append(col)

    # 1️Binary → Label Encoding
    for col in binary_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # 2️High Cardinality → Frequency Encoding
    for col in high_cardinality:
        freq = X[col].value_counts(normalize=True)
        X[col] = X[col].map(freq)

    # 3️Low Cardinality → OneHot Encoding
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), low_cardinality)
        ],
        remainder="drop"
    )

    X_processed = preprocessor.fit_transform(X)

    # Save preprocessor globally
    stored_preprocessor = preprocessor

    # Extract feature names properly
    feature_names = []

    # Numerical features (passthrough)
    feature_names.extend(num_features)

    # OneHot encoded features
    if low_cardinality:
        onehot_features = preprocessor.named_transformers_["onehot"] \
            .get_feature_names_out(low_cardinality)
        feature_names.extend(onehot_features.tolist())

    stored_feature_names = feature_names

    # Convert sparse matrix to dense 
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Store Processed Data
    stored_X = X_processed
    stored_y = y.values
    stored_problem_type = problem_type
    stored_target_column = target_column

    return JSONResponse(content={
        "original_shape": original_shape,
        "processed_feature_shape": X_processed.shape,
        "target_shape": y.shape,
        "problem_type": problem_type,
        "binary_encoded": binary_columns,
        "onehot_encoded": low_cardinality,
        "frequency_encoded": high_cardinality,
        "dropped_target_rows": dropped_rows,
        "message": "Preprocessing completed successfully"
    })


@app.post("/train/")
async def train_model():

    global stored_X, stored_y, stored_problem_type
    global stored_training_results, stored_best_model, stored_feature_names
    global stored_scaler, stored_needs_scaling

    if stored_X is None:
        return JSONResponse(content={"error": "Run preprocessing first"})

    X = stored_X
    y = stored_y
    problem_type = stored_problem_type
    results = {}

    # Split once for fair comparison
    if problem_type == "regression":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring_metric = "r2"

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring_metric = "accuracy"

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------
    # MODEL POOL (AutoML Core)
    # --------------------------

    if problem_type == "regression":

        models = {
            "LinearRegression": {"model": LinearRegression(), "scale": True},
            "Ridge": {"model": Ridge(), "scale": True},
            "Lasso": {"model": Lasso(), "scale": True},
            "ElasticNet": {"model": ElasticNet(), "scale": True},
            "RandomForest": {"model": RandomForestRegressor(random_state=42), "scale": False},
            "GradientBoosting": {"model": GradientBoostingRegressor(), "scale": False},
            "ExtraTrees": {"model": ExtraTreesRegressor(), "scale": False},
            "KNN": {"model": KNeighborsRegressor(), "scale": True},
            "SVR": {"model": SVR(), "scale": True}
        }

    else:

        models = {
            "LogisticRegression": {"model": LogisticRegression(max_iter=1000), "scale": True},
            "RandomForest": {"model": RandomForestClassifier(random_state=42), "scale": False},
            "GradientBoosting": {"model": GradientBoostingClassifier(), "scale": False},
            "ExtraTrees": {"model": ExtraTreesClassifier(), "scale": False},
            "KNN": {"model": KNeighborsClassifier(), "scale": True},
            "SVC": {"model": SVC(probability=True), "scale": True},
            "DecisionTree": {"model": DecisionTreeClassifier(), "scale": False},
            "NaiveBayes": {"model": GaussianNB(), "scale": False}
        }

    # --------------------------
    # MODEL EVALUATION LOOP
    # --------------------------

    for name, config in models.items():

        model = config["model"]
        needs_scaling = config["scale"]

        # Cross Validation
        X_cv = X_scaled if needs_scaling else X
        cv_scores = cross_val_score(model, X_cv, y, cv=cv_strategy, scoring=scoring_metric)
        cv_mean = cv_scores.mean()

        # Train-Test Evaluation
        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        if problem_type == "regression":
            test_score = r2_score(y_test, y_pred)
            results[name] = {
                "CV_R2_Mean": round(float(cv_mean), 4),
                "Test_R2": round(float(test_score), 4)
            }
        else:
            test_score = accuracy_score(y_test, y_pred)
            results[name] = {
                "CV_Accuracy_Mean": round(float(cv_mean), 4),
                "Test_Accuracy": round(float(test_score), 4)
            }

    # --------------------------
    # BEST MODEL SELECTION
    # --------------------------

    if problem_type == "regression":
        best_model_name = max(results, key=lambda x: results[x]["CV_R2_Mean"])
    else:
        best_model_name = max(results, key=lambda x: results[x]["CV_Accuracy_Mean"])

    best_model_config = models[best_model_name]
    final_model = best_model_config["model"]
    stored_needs_scaling = best_model_config["scale"]

    if stored_needs_scaling:
        stored_scaler = RobustScaler()
        X_scaled = stored_scaler.fit_transform(X)
        final_model.fit(X_scaled, y)
    else:
        stored_scaler = None
        final_model.fit(X, y)

    stored_best_model = final_model
    stored_training_results = results

    results["BestModel"] = best_model_name
    results["ProblemType"] = problem_type

    return JSONResponse(content=results)



def generate_statistical_insight(df, graph):

    insight = ""

    if graph["type"] == "histogram":

        col = graph["x"]
        data = df[col].dropna()

        if len(data) == 0:
            return "No sufficient data available for analysis."

        mean = data.mean()
        median = data.median()
        std = data.std()

        skew = data.skew()

        if skew > 0.5:
            shape = "right-skewed"
        elif skew < -0.5:
            shape = "left-skewed"
        else:
            shape = "fairly symmetric"

        insight = (
            f"The distribution of {col} has an average value of {mean:.2f} "
            f"with a median of {median:.2f}. The data appears {shape}, "
            f"indicating how values are concentrated across the dataset."
        )

    elif graph["type"] == "scatter":

        x = graph["x"]
        y = graph["y"]

        data = df[[x, y]].dropna()

        if len(data) == 0:
            return "Not enough data to evaluate relationship."

        corr = data[x].corr(data[y])

        if abs(corr) > 0.7:
            strength = "strong"
        elif abs(corr) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if corr > 0 else "negative"

        insight = (
            f"{x} and {y} show a {strength} {direction} correlation "
            f"({corr:.2f}). This suggests that changes in {x} "
            f"are associated with changes in {y}."
        )

    elif graph["type"] == "bar":

        col = graph["x"]

        counts = df[col].value_counts()

        if len(counts) == 0:
            return "No categorical distribution available."

        top = counts.idxmax()

        insight = (
            f"The category '{top}' appears most frequently in {col}, "
            f"indicating it dominates the dataset distribution."
        )

    elif graph["type"] == "box":

        x = graph["x"]
        y = graph["y"]

        groups = df.groupby(x)[y].mean().dropna()

        if len(groups) == 0:
            return "No group comparison insight available."

        top_group = groups.idxmax()

        insight = (
            f"The category '{top_group}' has the highest average {y}, "
            f"suggesting this group tends to produce larger values."
        )

    return insight

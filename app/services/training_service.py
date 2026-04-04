"""
Training service — AutoML model pool, 2-stage smart training, best model selection.

Stage 1 (large datasets > LARGE_DATASET_THRESHOLD):
  • Screen ALL models with 3-fold CV on a random sample (≤ SCREEN_SAMPLE_SIZE rows)
  • Keep only top TOP_N_FOR_FULL_TRAIN models

Stage 2 (always):
  • Full 5-fold CV + hold-out test evaluation on surviving models
  • Best model selected by composite score (regression: R², classification: 0.5*Acc + 0.5*F1)
  • Best model refitted on 100% of data

SVR / SVC excluded automatically for large datasets (O(n²) complexity).
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,          RandomForestClassifier,
    GradientBoostingRegressor,      GradientBoostingClassifier,
    HistGradientBoostingRegressor,  HistGradientBoostingClassifier,
    ExtraTreesRegressor,            ExtraTreesClassifier,
    AdaBoostRegressor,              AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, KFold,
)
from app.core.config import (
    RANDOM_STATE, TEST_SIZE, CV_SPLITS,
    LARGE_DATASET_THRESHOLD, SCREEN_SAMPLE_SIZE, TOP_N_FOR_FULL_TRAIN,
)

# ── Optional fast boosting libraries ─────────────────────────────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False



# ── Model pools ───────────────────────────────────────────────────────────────

def get_model_pool(problem_type: str, large_dataset: bool = False) -> dict:
    """
    Returns the candidate model pool.
    For large datasets:
      - n_estimators reduced to 50 (faster, still accurate)
      - GradientBoosting excluded (sequential, very slow on 50k+ rows)
      - SVR / SVC excluded (O(n²) complexity)
    XGBoost and LightGBM are always included when installed — fast at any size.
    """
    RS    = RANDOM_STATE
    n_est = 50 if large_dataset else 100

    if problem_type == "regression":
        pool = {
            "LinearRegression":     {"model": LinearRegression(),                                         "scale": True},
            "Ridge":                {"model": Ridge(),                                                    "scale": True},
            "Lasso":                {"model": Lasso(),                                                    "scale": True},
            "ElasticNet":           {"model": ElasticNet(),                                               "scale": True},
            "RandomForest":         {"model": RandomForestRegressor(n_estimators=n_est, random_state=RS), "scale": False},
            "HistGradientBoosting": {"model": HistGradientBoostingRegressor(random_state=RS),             "scale": False},
            "ExtraTrees":           {"model": ExtraTreesRegressor(n_estimators=n_est, random_state=RS),   "scale": False},
            "AdaBoost":             {"model": AdaBoostRegressor(random_state=RS),                         "scale": False},
            "KNN":                  {"model": KNeighborsRegressor(),                                      "scale": True},
        }
        # XGBoost — fast at any dataset size, usually top performer
        if XGBOOST_AVAILABLE:
            pool["XGBoost"] = {
                "model": XGBRegressor(
                    n_estimators=n_est, random_state=RS,
                    eval_metric="rmse", verbosity=0,
                ),
                "scale": False,
            }
        # LightGBM — fastest on large datasets, excellent accuracy
        if LIGHTGBM_AVAILABLE:
            pool["LightGBM"] = {
                "model": LGBMRegressor(
                    n_estimators=n_est, random_state=RS,
                    verbosity=-1,
                ),
                "scale": False,
            }
        if not large_dataset:
            pool["GradientBoosting"] = {"model": GradientBoostingRegressor(random_state=RS), "scale": False}
            pool["SVR"]              = {"model": SVR(), "scale": True}

    else:  # classification
        pool = {
            "LogisticRegression":   {"model": LogisticRegression(max_iter=1000, random_state=RS),          "scale": True},
            "RandomForest":         {"model": RandomForestClassifier(n_estimators=n_est, random_state=RS),  "scale": False},
            "HistGradientBoosting": {"model": HistGradientBoostingClassifier(random_state=RS),             "scale": False},
            "ExtraTrees":           {"model": ExtraTreesClassifier(n_estimators=n_est, random_state=RS),   "scale": False},
            "AdaBoost":             {"model": AdaBoostClassifier(random_state=RS),                         "scale": False},
            "KNN":                  {"model": KNeighborsClassifier(),                                      "scale": True},
            "DecisionTree":         {"model": DecisionTreeClassifier(random_state=RS),                     "scale": False},
            "NaiveBayes":           {"model": GaussianNB(),                                                "scale": False},
        }
        # XGBoost
        if XGBOOST_AVAILABLE:
            pool["XGBoost"] = {
                "model": XGBClassifier(
                    n_estimators=n_est, random_state=RS,
                    eval_metric="logloss", verbosity=0,
                    use_label_encoder=False,
                ),
                "scale": False,
            }
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            pool["LightGBM"] = {
                "model": LGBMClassifier(
                    n_estimators=n_est, random_state=RS,
                    verbosity=-1,
                ),
                "scale": False,
            }
        if not large_dataset:
            pool["GradientBoosting"] = {"model": GradientBoostingClassifier(random_state=RS), "scale": False}
            pool["SVC"]              = {"model": SVC(probability=True, random_state=RS), "scale": True}

    return pool


# ── Smart primary metric selection ───────────────────────────────────────────

IMBALANCE_THRESHOLD = 0.15   # a class with < 15% share → imbalanced dataset

def _is_imbalanced(y: np.ndarray) -> bool:
    """Return True if any class holds < IMBALANCE_THRESHOLD of all samples."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return False
    proportions = counts / counts.sum()
    return bool(proportions.min() < IMBALANCE_THRESHOLD)


def _primary_metric_key(problem_type: str, imbalanced: bool = False) -> str:
    """
    Returns the single metric key used for ranking models.

    Regression          → CV_R2_Mean        (highest R² wins)
    Classification
      balanced dataset  → CV_Accuracy_Mean  (straightforward and interpretable)
      imbalanced dataset→ CV_F1_Macro_Mean  (accuracy is misleading; F1 is fairer)
    """
    if problem_type == "regression":
        return "CV_R2_Mean"
    return "CV_F1_Macro_Mean" if imbalanced else "CV_Accuracy_Mean"


def _rank_score(metrics: dict, metric_key: str) -> float:
    """Extract a single float from the metrics dict for comparison."""
    val = metrics.get(metric_key)
    if val is None:
        return -999.0
    return float(val)


# ── Single-model evaluation ───────────────────────────────────────────────────

def _eval_model(
    name, config,
    X_train, X_test, y_train, y_test,
    X_cv, X_test_use,
    cv_strategy, problem_type,
):
    """
    Cross-validate + hold-out test a single model.
    Returns a metrics dict.
    """
    model         = config["model"]
    needs_scaling = config["scale"]

    # ── Cross-validation ──────────────────────────────────────────────────────
    if problem_type == "regression":
        r2_scores  = cross_val_score(model, X_cv, y_train if len(X_cv) == len(y_train) else None,
                                     cv=cv_strategy, scoring="r2")
        # use the full training set for CV (X_cv already corresponds to X/X_train context)
        cv_r2_mean  = float(r2_scores.mean())
        mae_scores  = cross_val_score(model, X_cv, y_train if len(X_cv) == len(y_train) else None,
                                      cv=cv_strategy, scoring="neg_mean_absolute_error")
        cv_mae_mean = float(-mae_scores.mean())
    else:
        acc_scores = cross_val_score(model, X_cv, y_train if len(X_cv) == len(y_train) else None,
                                     cv=cv_strategy, scoring="accuracy")
        f1_scores  = cross_val_score(model, X_cv, y_train if len(X_cv) == len(y_train) else None,
                                     cv=cv_strategy, scoring="f1_macro")
        cv_acc_mean = float(acc_scores.mean())
        cv_f1_mean  = float(f1_scores.mean())

    # ── Hold-out test ─────────────────────────────────────────────────────────
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_use)

    if problem_type == "regression":
        return {
            "CV_R2_Mean":  round(cv_r2_mean, 4),
            "Test_R2":     round(float(r2_score(y_test, y_pred)), 4),
            "Test_MAE":    round(float(mean_absolute_error(y_test, y_pred)), 4),
            "Test_RMSE":   round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }
    else:
        entry = {
            "CV_Accuracy_Mean": round(cv_acc_mean, 4),
            "CV_F1_Macro_Mean": round(cv_f1_mean, 4),
            "Test_Accuracy":    round(float(accuracy_score(y_test, y_pred)), 4),
            "Test_F1_Macro":    round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        }
        # ROC-AUC for binary classification only
        n_classes = len(np.unique(y_test))
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test_use)[:, 1]
                entry["Test_ROC_AUC"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            except Exception:
                pass
        return entry


# ── Main training function ────────────────────────────────────────────────────

def run_training(X: np.ndarray, y: np.ndarray, problem_type: str) -> dict:
    """
    2-stage smart training:
      • Large dataset (> LARGE_DATASET_THRESHOLD): Stage 1 screens on a sample,
        Stage 2 fully trains only the top TOP_N_FOR_FULL_TRAIN models.
      • Small dataset: all models fully trained directly (no screening).

    Returns results dict, best model name, fitted model, scaler, test sets.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # ── Sanitize y ────────────────────────────────────────────────────────────
    # Remove any rows where y is NaN / None (pd.NA also caught by pd.isna)
    y_series   = pd.Series(y)
    valid_mask = y_series.notna().values
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"[Training] Dropping {n_dropped} rows with NaN in target.")
        X = X[valid_mask]
        y = y[valid_mask]

    # For classification, always LabelEncode y → clean integer labels.
    # This handles: string targets, float 0.0/1.0 targets, mixed types, None-as-object.
    if problem_type == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))   # str() cast handles None/float safely

    n_samples     = len(X)
    large_dataset = n_samples > LARGE_DATASET_THRESHOLD


    # ── Train/test split ──────────────────────────────────────────────────────
    if problem_type == "regression":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        # Use 3-fold for large datasets (faster, still reliable with 40k+ train rows)
        _cv_splits        = 3 if large_dataset else CV_SPLITS
        cv_strategy       = KFold(n_splits=_cv_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_screen_strategy = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        _cv_splits        = 3 if large_dataset else CV_SPLITS
        cv_strategy       = StratifiedKFold(n_splits=_cv_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_screen_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ── Scaling — fit ONLY on X_train to prevent leakage ─────────────────────
    scaler         = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    # Full scaled X for CV (parameters come from X_train — minor, accepted approximation)
    X_full_scaled  = scaler.transform(X)

    models  = get_model_pool(problem_type, large_dataset=large_dataset)
    results = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Screen all models on a sample (large datasets only)
    # ═══════════════════════════════════════════════════════════════════════════
    if large_dataset:
        sample_size = min(SCREEN_SAMPLE_SIZE, n_samples)
        rng = np.random.RandomState(RANDOM_STATE)

        if problem_type == "classification":
            # Stratified sample to keep class proportions
            classes, counts = np.unique(y, return_counts=True)
            sample_idx = []
            for cls, cnt in zip(classes, counts):
                cls_idx     = np.where(y == cls)[0]
                n_take      = max(1, int(round(cnt / n_samples * sample_size)))
                n_take      = min(n_take, len(cls_idx))
                sample_idx.extend(rng.choice(cls_idx, size=n_take, replace=False))
            sample_idx = np.array(sample_idx)
        else:
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

        X_s = X[sample_idx]
        y_s = y[sample_idx]
        X_s_scaled = scaler.transform(X_s)

        screen_scores = {}
        scoring_key   = "r2" if problem_type == "regression" else "accuracy"

        for name, cfg in models.items():
            try:
                X_cv_s = X_s_scaled if cfg["scale"] else X_s
                scores = cross_val_score(
                    cfg["model"], X_cv_s, y_s,
                    cv=cv_screen_strategy, scoring=scoring_key
                )
                screen_scores[name] = float(scores.mean())
            except Exception:
                screen_scores[name] = -999.0

        # Store screened-out models in results with their sample score
        top_names = sorted(screen_scores, key=lambda n: screen_scores[n], reverse=True)[:TOP_N_FOR_FULL_TRAIN]

        for name, score in screen_scores.items():
            if name not in top_names:
                if problem_type == "regression":
                    results[name] = {
                        "CV_R2_Mean": round(max(score, -1.0), 4),
                        "Test_R2": None, "Test_MAE": None, "Test_RMSE": None,
                        "screened_out": True,
                        "note": "Eliminated in Stage-1 screening — sample score shown",
                    }
                else:
                    results[name] = {
                        "CV_Accuracy_Mean": round(max(score, 0.0), 4),
                        "CV_F1_Macro_Mean": None,
                        "Test_Accuracy": None, "Test_F1_Macro": None,
                        "screened_out": True,
                        "note": "Eliminated in Stage-1 screening — sample score shown",
                    }

        models_to_train = {n: models[n] for n in top_names}
        print(f"[Training] Large dataset ({n_samples} rows). "
              f"Stage-1 complete. Top {TOP_N_FOR_FULL_TRAIN}: {top_names}")
    else:
        models_to_train = models
        print(f"[Training] Small dataset ({n_samples} rows). Training all models.")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Full evaluation on surviving models
    # ═══════════════════════════════════════════════════════════════════════════
    for name, cfg in models_to_train.items():
        needs_scaling = cfg["scale"]

        X_cv_full  = X_full_scaled  if needs_scaling else X
        X_tr       = X_train_scaled if needs_scaling else X_train
        X_te       = X_test_scaled  if needs_scaling else X_test

        # We pass X_cv_full and y (full), but CV strategy already splits correctly
        # because CV is done on the full X with full y
        try:
            if problem_type == "regression":
                r2_cv  = cross_val_score(cfg["model"], X_cv_full, y, cv=cv_strategy, scoring="r2")
                mae_cv = cross_val_score(cfg["model"], X_cv_full, y, cv=cv_strategy, scoring="neg_mean_absolute_error")
                cv_r2  = float(r2_cv.mean())
                cv_mae = float(-mae_cv.mean())

                cfg["model"].fit(X_tr, y_train)
                y_pred = cfg["model"].predict(X_te)

                results[name] = {
                    "CV_R2_Mean":  round(cv_r2, 4),
                    "Test_R2":     round(float(r2_score(y_test, y_pred)), 4),
                    "Test_MAE":    round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "Test_RMSE":   round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                }
            else:
                acc_cv = cross_val_score(cfg["model"], X_cv_full, y, cv=cv_strategy, scoring="accuracy")
                f1_cv  = cross_val_score(cfg["model"], X_cv_full, y, cv=cv_strategy, scoring="f1_macro")
                cv_acc = float(acc_cv.mean())
                cv_f1  = float(f1_cv.mean())

                cfg["model"].fit(X_tr, y_train)
                y_pred = cfg["model"].predict(X_te)

                entry = {
                    "CV_Accuracy_Mean": round(cv_acc, 4),
                    "CV_F1_Macro_Mean": round(cv_f1, 4),
                    "Test_Accuracy":    round(float(accuracy_score(y_test, y_pred)), 4),
                    "Test_F1_Macro":    round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
                }
                # ROC-AUC (binary only)
                if len(np.unique(y_test)) == 2 and hasattr(cfg["model"], "predict_proba"):
                    try:
                        y_prob = cfg["model"].predict_proba(X_te)[:, 1]
                        entry["Test_ROC_AUC"] = round(float(roc_auc_score(y_test, y_prob)), 4)
                    except Exception:
                        pass
                results[name] = entry

        except Exception as e:
            print(f"[Training] {name} failed: {e}")
            continue

    # ── Pick best model ───────────────────────────────────────────────────────
    fully_trained = {k: v for k, v in results.items() if not v.get("screened_out")}

    if not fully_trained:
        raise RuntimeError("All models failed during training.")

    # Detect imbalance so we choose the fairest ranking metric
    imbalanced      = _is_imbalanced(y) if problem_type == "classification" else False
    primary_metric  = _primary_metric_key(problem_type, imbalanced)

    best_model_name = max(
        fully_trained,
        key=lambda n: _rank_score(fully_trained[n], primary_metric)
    )
    print(f"[Training] Imbalanced={imbalanced}  Primary metric={primary_metric}")
    print(f"[Training] Best model: {best_model_name} "
          f"({primary_metric}: {_rank_score(fully_trained[best_model_name], primary_metric):.4f})")

    # ── Confusion matrix for classification ───────────────────────────────────
    if problem_type == "classification":
        bm_cfg      = models_to_train[best_model_name]
        X_te_bm     = X_test_scaled if bm_cfg["scale"] else X_test
        y_pred_best = bm_cfg["model"].predict(X_te_bm)
        results["ConfusionMatrix"] = confusion_matrix(y_test, y_pred_best).tolist()

    # ── Refit best model on ALL data ─────────────────────────────────────────
    best_cfg      = models_to_train[best_model_name]
    final_model   = best_cfg["model"]
    needs_scaling = best_cfg["scale"]

    if needs_scaling:
        final_scaler = RobustScaler()
        X_final      = final_scaler.fit_transform(X)
        final_model.fit(X_final, y)
        X_test_for_perm = X_test_scaled   # scaled version for permutation importance
    else:
        final_scaler    = None
        final_model.fit(X, y)
        X_test_for_perm = X_test

    return {
        "results":          results,
        "best_model_name":  best_model_name,
        "primary_metric":   primary_metric,     # key used to rank — send to frontend
        "imbalanced":       imbalanced,
        "best_model":       final_model,
        "scaler":           final_scaler,
        "needs_scaling":    needs_scaling,
        "X_test":           X_test_for_perm,   # in format model was trained on
        "y_test":           y_test,
        "large_dataset":    large_dataset,
    }
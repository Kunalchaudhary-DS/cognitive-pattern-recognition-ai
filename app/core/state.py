# Central session state — all routes and services read/write through this object.

import pandas as pd
import numpy as np
from typing import Optional


class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.problem_type: Optional[str] = None          # "classification" | "regression"
        self.target_column: Optional[str] = None
        self.feature_names: Optional[list] = None
        self.preprocessor = None                          # sklearn ColumnTransformer
        self.best_model = None                            # fitted sklearn model
        self.training_results: Optional[dict] = None
        self.scaler = None                                # RobustScaler if needed
        self.needs_scaling: bool = False
        self.strong_correlations: list = []
        self.encoding_maps: dict = {}
        self.X_test: Optional[np.ndarray] = None         # held-out test features (model-format)
        self.y_test: Optional[np.ndarray] = None         # held-out test labels
        self.label_encoder = None                         # LabelEncoder for classification y (inverse-transform predictions → original labels)
        self.constraint_map: Optional[dict] = None        # merged constraint rulebook (Semantic Prediction Interceptor)

    def reset(self):
        """Clear all state — useful for starting fresh with a new dataset."""
        self.__init__()


# Singleton shared by all routers and services
state = AppState()
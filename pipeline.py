import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Average temperature
        X["avg_temp"] = (X["maxtemp"] + X["mintemp"]) / 2

        # Season from day
        def assign_season(day):
            if day <= 90:
                return "winter"
            elif day <= 180:
                return "spring"
            elif day <= 270:
                return "summer"
            else:
                return "fall"

        X["season"] = X["day"].apply(assign_season)
        X = pd.get_dummies(X, columns=["season"], drop_first=True)

        # Log transform skewed features
        for col in ["humidity", "cloud", "dewpoint"]:
            X[col + "_log"] = np.log1p(X[col])

        # Drop unused columns
        X.drop(columns=["id", "day", "maxtemp", "mintemp", "temparature", 
                        "humidity", "cloud", "dewpoint"], inplace=True)
        return X

def build_pipeline(model=None):
    return ImbPipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("model", model if model else RandomForestClassifier(n_estimators=100, random_state=42))
    ])

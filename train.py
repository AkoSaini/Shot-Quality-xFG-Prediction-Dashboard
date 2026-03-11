import os
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss


DATA_PATH = os.path.join("data", "shots.csv")
MODEL_PATH = "model.joblib"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Kobe dataset uses loc_x, loc_y
    x = pd.to_numeric(df["loc_x"], errors="coerce")
    y = pd.to_numeric(df["loc_y"], errors="coerce")

    df["distance_xy"] = np.sqrt(x**2 + y**2)
    df["angle"] = np.arctan2(y, x)

    # Time remaining in quarter (seconds)
    df["time_remaining_sec"] = (
        pd.to_numeric(df["minutes_remaining"], errors="coerce") * 60
        + pd.to_numeric(df["seconds_remaining"], errors="coerce")
    )

    return df


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Put your CSV at data/shots.csv")

    df = pd.read_csv(DATA_PATH)

    # Kobe dataset has missing shot_made_flag for some rows
    df = df[df["shot_made_flag"].notna()].copy()

    y = df["shot_made_flag"].astype(int)
    X = df.drop(columns=["shot_made_flag"])

    X = add_features(X)

    # Compact set of columns that work well and train fast
    num_cols = [
        "shot_distance", "distance_xy", "angle",
        "period", "playoffs", "time_remaining_sec"
    ]
    cat_cols = [
        "combined_shot_type",
        "shot_type",
        "shot_zone_basic",
        "shot_zone_area",
        "shot_zone_range",
        "opponent",
        "season",
    ]

    # Only keep columns that exist 
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

    X = X[num_cols + cat_cols].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

   #  model = LogisticRegression(max_iter=1000, solver="saga")
   # Gradient boosting performs better than logistic regression for this dataset, and trains fast enough with histograms.
    model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
    )


    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    p = pipe.predict_proba(X_test)[:, 1]
    print(f"Test ROC-AUC: {roc_auc_score(y_test, p):.4f}")
    print(f"Test LogLoss: {log_loss(y_test, p):.4f}")

    dump(pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()

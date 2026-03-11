import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

DATA_PATH = os.path.join("data", "shots.csv")
MODEL_PATH = "model.joblib"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    x = pd.to_numeric(df["loc_x"], errors="coerce")
    y = pd.to_numeric(df["loc_y"], errors="coerce")

    df["distance_xy"] = np.sqrt(x**2 + y**2)
    df["angle"] = np.arctan2(y, x)
    df["time_remaining_sec"] = (
        pd.to_numeric(df["minutes_remaining"], errors="coerce") * 60
        + pd.to_numeric(df["seconds_remaining"], errors="coerce")
    )
    return df


st.set_page_config(page_title="Shot Quality (xFG)", layout="wide")
st.title("Shot Quality (xFG) Dashboard — Kobe Dataset")

if not os.path.exists(DATA_PATH):
    st.error("Missing data/shots.csv. Put your CSV there and reload.")
    st.stop()

# Auto-train if model missing
if not os.path.exists(MODEL_PATH):
    st.warning("model.joblib not found — training now...")
    subprocess.check_call([sys.executable, "train.py"])

model = load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df = add_features(df)

# Sidebar filters
st.sidebar.header("Filters")
if "season" in df.columns:
    season = st.sidebar.selectbox("Season", ["All"] + sorted(df["season"].dropna().unique().tolist()))
    if season != "All":
        df = df[df["season"] == season]

if "shot_zone_basic" in df.columns:
    zone = st.sidebar.selectbox("Zone", ["All"] + sorted(df["shot_zone_basic"].dropna().unique().tolist()))
    if zone != "All":
        df = df[df["shot_zone_basic"] == zone]

if "opponent" in df.columns:
    opp = st.sidebar.selectbox("Opponent", ["All"] + sorted(df["opponent"].dropna().unique().tolist()))
    if opp != "All":
        df = df[df["opponent"] == opp]

# Build X exactly like training did
num_cols = ["shot_distance", "distance_xy", "angle", "period", "playoffs", "time_remaining_sec"]
cat_cols = ["combined_shot_type", "shot_type", "shot_zone_basic", "shot_zone_area", "shot_zone_range", "opponent", "season"]

num_cols = [c for c in num_cols if c in df.columns]
cat_cols = [c for c in cat_cols if c in df.columns]

X = df[num_cols + cat_cols].copy()
df["xFG"] = model.predict_proba(X)[:, 1]

# Metrics
labeled = df[df["shot_made_flag"].notna()].copy()
c1, c2, c3 = st.columns(3)
c1.metric("Shots (filtered)", len(df))
c2.metric("Expected FG% (xFG)", f"{df['xFG'].mean():.3f}")

if len(labeled) > 0:
    c3.metric("Actual FG% (labeled only)", f"{labeled['shot_made_flag'].mean():.3f}")
else:
    c3.metric("Actual FG% (labeled only)", "N/A")

st.subheader("Shot Map")
st.scatter_chart(df, x="loc_x", y="loc_y")

st.subheader("Top rows")
show_cols = ["season", "opponent", "period", "shot_distance", "shot_zone_basic", "shot_made_flag", "xFG"]
show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(df[show_cols].head(50))

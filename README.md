# Shot Quality (xFG) Prediction & Dashboard

This project predicts basketball shot make probability (expected field goal percentage, **xFG**) from shot location and game context, then visualizes results in an interactive **Streamlit** dashboard. It uses the **Kobe Bryant Shot Selection** dataset containing shot attempts across his career.

## Why this project
Shot outcomes are noisy, but shot *quality* is more stable. By estimating `P(make)` for each attempt, the model enables analysis like:
- Comparing **expected vs actual FG%** by zone, season, or opponent
- Identifying which shot locations/types are generally higher quality
- Exploring how shot context affects make probability

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- Streamlit
- joblib (model persistence)

## Results
Model: **HistGradientBoostingClassifier**  
Evaluation (held-out test set): **ROC-AUC 0.630**, **LogLoss 0.650**

**What the metrics mean**
- **ROC-AUC** measures how well the model ranks makes above misses (0.5 = random, 1.0 = perfect).
- **LogLoss** measures probability accuracy (lower is better; wrong confident predictions are penalized heavily).

## Dataset
**Kobe Bryant Shot Selection (Kaggle)**  
Target label: `shot_made_flag` (1 = made, 0 = missed). Rows with missing labels are excluded from training.

### Dataset setup
1. Download the dataset from Kaggle.
2. Place the CSV at: `data/shots.csv`

## Features used by the model
### Engineered features
Derived from shot coordinates and clock:
- `distance_xy = sqrt(loc_x^2 + loc_y^2)`
- `angle = atan2(loc_y, loc_x)`
- `time_remaining_sec = minutes_remaining*60 + seconds_remaining`

### Context features
Used directly from the dataset:
- Numeric: `shot_distance`, `period`, `playoffs`
- Categorical (one-hot encoded): `combined_shot_type`, `shot_type`, `shot_zone_basic`, `shot_zone_area`, `shot_zone_range`, `opponent`, `season`

## How it works
1. `train.py` loads `data/shots.csv`, engineers features, preprocesses data, trains the model, prints evaluation metrics, and saves `model.joblib`.
2. `app.py` loads `model.joblib`, computes xFG for filtered shots, and visualizes results in Streamlit.

## Run locally
```bash
pip install -r requirements.txt
python train.py
python -m streamlit run app.py

# Shot Quality (xFG) Prediction & Dashboard

Predicts shot make probability (xFG) using shot location + context and visualizes results in a Streamlit dashboard. Uses a dataset of all the shots taken by Kobe Bryant in his career.

## Tech
Python, pandas, scikit-learn, Streamlit

## Results
Gradient Boosting: ROC-AUC 0.630, LogLoss 0.650 (held-out test set)

## Run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py

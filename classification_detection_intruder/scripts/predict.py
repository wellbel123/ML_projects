import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pickle
from pathlib import Path

from src.features import extract_features
from src.vectorizer import sessions_to_text, load_vectorizer, transform_text
from src.modeling import evaluate_model
import joblib
from scipy.sparse import hstack

# ---------- Config ----------
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
MODEL_NAME = "alice"

TEST_FILE = DATA_DIR / "test_sessions.csv"
TRAIN_FILE = DATA_DIR / "train_sessions.csv"
DICT_FILE = DATA_DIR / "site_dic.pkl"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

# ---------- Constants ----------
times = [f"time{i}" for i in range(1, 11)]
sites = [f"site{i}" for i in range(1, 11)]

# ---------- Load model components ----------
print("Loading model components...")
model = joblib.load(OUTPUT_DIR / f"{MODEL_NAME}_model.pkl")
scaler = joblib.load(OUTPUT_DIR / f"{MODEL_NAME}_scaler.pkl")
vectorizer = joblib.load(OUTPUT_DIR / f"{MODEL_NAME}_vectorizer.pkl")

# ---------- Load site dictionary ----------
with open(DICT_FILE, "rb") as f:
    site_dic = pickle.load(f)
id2site = {v: k for k, v in site_dic.items()}
id2site[0] = "unknown"

# ---------- Load train data to extract top-10 sets ----------
print("Loading top-10 sets...")
with open(OUTPUT_DIR / f"{MODEL_NAME}_top10.pkl", "rb") as f:
    top_sets = pickle.load(f)
top_10_set = top_sets["top_10_set"]
top_10_alice_set = top_sets["top_10_alice_set"]

# ---------- Load test data ----------
print("Loading and processing test data...")
test_df = pd.read_csv(TEST_FILE, index_col="session_id", parse_dates=times)

test_df = extract_features(test_df, id2site, top_10_set, top_10_alice_set)

# ---------- Convert sessions to text and transform ----------
test_text = sessions_to_text(test_df, id2site)
X_test_sparse = transform_text(vectorizer, test_text)

# ---------- Select features ----------
feature_columns = [
    'has_top_10', 'has_top_10_Alice', 'top_10_count',
    'session_steps', 'uniq_sites', 'top_alices_sites_share',
    'session_duration', 'working_hours', 'hour_sin_x',
    'hour_cos_x', 'day_of_week', 'is_weekend', 'month', 'day'
]
X_test_feats = test_df[feature_columns]
X_test_scaled = scaler.transform(X_test_feats)

# ---------- Combine and predict ----------
X_test_final = hstack([X_test_sparse, X_test_scaled])
y_pred_proba = model.predict_proba(X_test_final)[:, 1]

# ---------- Save submission ----------
print("Saving submission file...")
submission = pd.DataFrame({
    "session_id": test_df.index,
    "target": y_pred_proba
})
submission.to_csv(SUBMISSION_FILE, index=False)
print("Submission saved to:", SUBMISSION_FILE)

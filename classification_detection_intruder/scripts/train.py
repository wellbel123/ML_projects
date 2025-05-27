import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pickle

import yaml


from src.features import extract_features
from src.vectorizer import sessions_to_text, fit_vectorizer
from src.pipeline import build_model_pipeline
from src.modeling import train_model, evaluate_model, save_model

with open("config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

C = params["model"]["C"]
ngram_range = tuple(params["vectorizer"]["ngram_range"])
max_features = params["vectorizer"]["max_features"]

# ---------- Config ----------
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
MODEL_NAME = "alice"

TRAIN_FILE = DATA_DIR / "train_sessions.csv"
DICT_FILE = DATA_DIR / "site_dic.pkl"

# ---------- Load data ----------
print("Loading data...")
times = [f"time{i}" for i in range(1, 11)]
sites = [f"site{i}" for i in range(1, 11)]

train_df = pd.read_csv(TRAIN_FILE, index_col="session_id", parse_dates=times)

with open(DICT_FILE, "rb") as f:
    site_dic = pickle.load(f)
id2site = {v: k for k, v in site_dic.items()}
id2site[0] = "unknown"

# ---------- Feature engineering ----------
print("Generating features...")
top_sites = pd.Series(train_df[sites].values.flatten()).value_counts().sort_values(ascending=False)
top_10_set = set(top_sites.head(10).index)

top_sites_alice = pd.Series(train_df[train_df["target"] == 1][sites].values.flatten()).value_counts().sort_values(ascending=False)
top_10_alice_set = set(top_sites_alice.head(10).index)

train_df = extract_features(train_df, id2site, top_10_set, top_10_alice_set)

# ---------- Convert sessions to text ----------
print("Converting sessions to text...")
session_texts = sessions_to_text(train_df, id2site)
targets = train_df["target"]

# ---------- Time-based split ----------
print("Splitting by time...")
split_index = int(len(session_texts) * 0.8)

train_text = session_texts[:split_index]
valid_text = session_texts[split_index:]

y_train = targets.iloc[:split_index]
y_valid = targets.iloc[split_index:]

train_features_df = train_df.iloc[:split_index]
valid_features_df = train_df.iloc[split_index:]

# ---------- Vectorize ----------
print("Vectorizing...")
vectorizer, X_train_sparse = fit_vectorizer(train_text)
X_valid_sparse = vectorizer.transform(valid_text)

# ---------- Feature selection ----------
feature_columns = [
    'has_top_10', 'has_top_10_Alice', 'top_10_count',
    'session_steps', 'uniq_sites', 'top_alices_sites_share',
    'session_duration', 'working_hours', 'hour_sin_x',
    'hour_cos_x', 'day_of_week', 'is_weekend', 'month', 'day'
]

X_train_feats = train_features_df[feature_columns]
X_valid_feats = valid_features_df[feature_columns]

# ---------- Build model ----------
print("Building model pipeline...")
pipeline_parts = build_model_pipeline(vectorizer, C=C)
scaler = pipeline_parts["scaler"]
classifier = pipeline_parts["classifier"]

# ---------- Train ----------
print("Training model...")
classifier, scaler = train_model(classifier, X_train_sparse, X_train_feats, y_train, scaler)

# ---------- Evaluate ----------
print("Evaluating...")
roc_auc = evaluate_model(classifier, X_valid_sparse, X_valid_feats, y_valid, scaler)
print(f"Validation ROC AUC: {roc_auc:.4f}")

# ---------- Save model ----------
print("Saving model...")
OUTPUT_DIR.mkdir(exist_ok=True)
save_model(classifier, scaler, vectorizer, OUTPUT_DIR / MODEL_NAME)
print("Done.")

# ---------- Save top-10 sets ----------
print("Saving top-10 sets...")
with open(OUTPUT_DIR / f"{MODEL_NAME}_top10.pkl", "wb") as f:
    pickle.dump({
        "top_10_set": top_10_set,
        "top_10_alice_set": top_10_alice_set
    }, f)
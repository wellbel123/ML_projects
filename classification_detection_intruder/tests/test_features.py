import pandas as pd
from src.features import extract_features

def test_extract_features_creates_expected_columns():
    df = pd.DataFrame({
        "site1": [1], "site2": [2], "site3": [0], "site4": [0], "site5": [0],
        "site6": [0], "site7": [0], "site8": [0], "site9": [0], "site10": [0],
        "time1": ["2023-01-01 09:00:00"], "time2": ["2023-01-01 09:00:10"], 
        **{f"time{i}": [None] for i in range(3, 11)}
    })

    df["time1"] = pd.to_datetime(df["time1"])
    df["time2"] = pd.to_datetime(df["time2"])

    id2site = {1: "google.com", 2: "facebook.com", 0: "unknown"}
    top10 = {1, 2}
    df = extract_features(df, id2site, top10, top10)

    expected_columns = [
        'session_steps', 'uniq_sites', 'session_duration',
        'start_hour', 'working_hours', 'hour_sin_x', 'hour_cos_x',
        'day_of_week', 'is_weekend', 'month', 'day',
        'has_top_10', 'top_10_count', 'has_top_10_Alice',
        'top_10_Alice_count', 'top_alices_sites_share'
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

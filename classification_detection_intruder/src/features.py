import numpy as np
import pandas as pd

SITES = [f"site{i}" for i in range(1, 11)]
TIMES = [f"time{i}" for i in range(1, 11)]
PI = np.pi

def extract_features(df, id2site, top_10_set=None, top_10_alice_set=None):
    # ensure time columns are in datetime
    df[TIMES] = df[TIMES].apply(pd.to_datetime, errors='coerce')
    df[SITES] = df[SITES].fillna(0).astype(int)

    df['session_steps'] = df[SITES].apply(lambda x: (x != 0).sum(), axis=1)
    df['uniq_sites'] = df[SITES].apply(lambda x: x[x != 0].nunique(), axis=1)
    
    df['min_time'] = df[TIMES].min(axis=1)
    df['max_time'] = df[TIMES].max(axis=1)
    df['session_duration'] = (df['max_time'] - df['min_time']) / np.timedelta64(1, 's')
    
    df['start_hour'] = df['min_time'].dt.hour
    df['working_hours'] = df['start_hour'].apply(lambda x: int(9 <= x <= 18))
    df['hour_sin_x'] = np.sin(2 * PI * df['start_hour'] / 24)
    df['hour_cos_x'] = np.cos(2 * PI * df['start_hour'] / 24)
    
    df['day_of_week'] = df['min_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['min_time'].dt.month
    df['day'] = df['min_time'].dt.day

    # top-10 features — optional
    if top_10_set is not None:
        df['has_top_10'] = df[SITES].apply(lambda x: int(any(site in top_10_set for site in x)), axis=1)
        df['top_10_count'] = df[SITES].apply(lambda x: sum(site in top_10_set for site in x), axis=1)
    else:
        df['has_top_10'] = 0
        df['top_10_count'] = 0

    if top_10_alice_set is not None:
        df['has_top_10_Alice'] = df[SITES].apply(lambda x: int(any(site in top_10_alice_set for site in x)), axis=1)
        df['top_10_Alice_count'] = df[SITES].apply(lambda x: sum(site in top_10_alice_set for site in x), axis=1)
        df['top_alices_sites_share'] = df['top_10_Alice_count'] / df['session_steps'].replace(0, 1)
    else:
        df['has_top_10_Alice'] = 0
        df['top_10_Alice_count'] = 0
        df['top_alices_sites_share'] = 0

    return df
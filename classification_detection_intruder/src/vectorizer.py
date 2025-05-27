import joblib
from sklearn.feature_extraction.text import CountVectorizer

SITES = [f"site{i}" for i in range(1, 11)]

def sessions_to_text(df, id2site):
    return df[SITES].apply(lambda row: ' '.join([id2site.get(site, 'unknown') for site in row]), axis=1).tolist()

def get_vectorizer(ngram_range=(1, 3), max_features=50000):
    return CountVectorizer(ngram_range=ngram_range, max_features=max_features)

def fit_vectorizer(text_data, ngram_range=(1, 3), max_features=50000):
    vectorizer = get_vectorizer(ngram_range, max_features)
    X = vectorizer.fit_transform(text_data)
    return vectorizer, X

def transform_text(vectorizer, text_data):
    return vectorizer.transform(text_data)

def save_vectorizer(vectorizer, filepath):
    joblib.dump(vectorizer, filepath)

def load_vectorizer(filepath):
    return joblib.load(filepath)
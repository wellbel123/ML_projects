from src.vectorizer import sessions_to_text, fit_vectorizer
import pandas as pd

def test_sessions_to_text_produces_expected_format():
    df = pd.DataFrame({
        "site1": [1], "site2": [2], "site3": [0], "site4": [0], "site5": [0],
        "site6": [0], "site7": [0], "site8": [0], "site9": [0], "site10": [0],
    })
    id2site = {0: "unknown", 1: "google.com", 2: "facebook.com"}

    text = sessions_to_text(df, id2site)
    assert isinstance(text, list)
    assert text[0] == "google.com facebook.com unknown unknown unknown unknown unknown unknown unknown unknown"

def test_vectorizer_learns_vocabulary():
    texts = ["google.com facebook.com", "youtube.com google.com"]
    vectorizer, X = fit_vectorizer(texts, ngram_range=(1, 2), max_features=10)
    vocab = vectorizer.vocabulary_
    assert "google.com" in vocab
    assert "google.com facebook.com" in vocab or "facebook.com google.com" in vocab
    assert X.shape[0] == 2

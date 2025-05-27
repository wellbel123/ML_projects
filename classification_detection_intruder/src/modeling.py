import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

def train_model(classifier, X_sparse, X_dense, y, scaler=None):
    """
    Trains a classifier using combined sparse (text) and dense (numeric) features.

    Parameters:
        classifier: model instance (e.g., LogisticRegression)
        X_sparse: sparse matrix from CountVectorizer
        X_dense: numeric features (DataFrame or array)
        y: target labels
        scaler: StandardScaler instance (optional)

    Returns:
        Trained classifier and scaled numeric features
    """
    if scaler:
        X_dense = scaler.fit_transform(X_dense)
    X_combined = hstack([X_sparse, X_dense])
    classifier.fit(X_combined, y)
    return classifier, scaler

def evaluate_model(classifier, X_sparse, X_dense, y_true, scaler=None):
    """
    Evaluates classifier using ROC-AUC on given data.

    Parameters:
        classifier: trained model
        X_sparse: sparse matrix from CountVectorizer
        X_dense: numeric features
        y_true: true target values
        scaler: StandardScaler instance (optional)

    Returns:
        ROC-AUC score
    """
    if scaler:
        X_dense = scaler.transform(X_dense)
    X_combined = hstack([X_sparse, X_dense])
    y_pred = classifier.predict_proba(X_combined)[:, 1]
    return roc_auc_score(y_true, y_pred)

def save_model(classifier, scaler, vectorizer, filepath_prefix):
    """
    Saves model components (classifier, scaler, vectorizer) to disk.

    Parameters:
        classifier: trained model
        scaler: fitted StandardScaler
        vectorizer: fitted CountVectorizer
        filepath_prefix: prefix path for saving files

    Output:
        Saves three files:
            <prefix>_model.pkl
            <prefix>_scaler.pkl
            <prefix>_vectorizer.pkl
    """
    joblib.dump(classifier, f"{filepath_prefix}_model.pkl")
    joblib.dump(scaler, f"{filepath_prefix}_scaler.pkl")
    joblib.dump(vectorizer, f"{filepath_prefix}_vectorizer.pkl")

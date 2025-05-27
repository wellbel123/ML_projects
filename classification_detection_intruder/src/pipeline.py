from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

def build_model_pipeline(vectorizer, C=1.0):
    """
    Builds a pipeline that combines:
    - a trained CountVectorizer for textual session data
    - scaled numeric features (hand-crafted)
    - logistic regression as the final classifier

    Note: Since we're dealing with two separate data sources 
    (sparse matrix from vectorizer + dense matrix from features),
    this function assumes external handling of combination (e.g., via hstack).

    Parameters:
        vectorizer: trained CountVectorizer
        C: regularization strength for logistic regression

    Returns:
        A dictionary with:
            - 'vectorizer': CountVectorizer (trained)
            - 'scaler': StandardScaler (not fitted)
            - 'classifier': LogisticRegression (not fitted)
    """
    scaler = StandardScaler()
    clf = LogisticRegression(C=C, random_state=17, solver='liblinear')

    return {
        'vectorizer': vectorizer,
        'scaler': scaler,
        'classifier': clf
    }

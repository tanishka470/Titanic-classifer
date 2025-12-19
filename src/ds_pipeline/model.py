from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])

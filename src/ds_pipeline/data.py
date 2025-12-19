import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing: {missing}")
    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


def make_splits(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    X, y = split_features_labels(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
CATEGORICAL = ["Sex", "Embarked"]


def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL),
    ])

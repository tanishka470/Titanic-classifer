import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ds_pipeline.data import load_data, make_splits
from ds_pipeline.features import build_preprocessor
from ds_pipeline.model import build_model


def test_pipeline_smoke():
    df = load_data(str(PROJECT_ROOT / "data" / "titanic_sample.csv"))
    X_train, X_test, y_train, y_test = make_splits(df, test_size=0.25, random_state=0)
    pipeline = build_model(build_preprocessor())
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)

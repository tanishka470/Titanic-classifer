from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    report = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            report["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            report["roc_auc"] = float("nan")
    return report

"""features_temiz için sentetik testler. Çalıştırma: python tests/test_features_temiz.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import features_temiz as FT  # noqa: E402


def test_matrices():
    tr = pd.DataFrame({"ADES_mvt": ["LTBA"] * 60 + ["KJFK"] * 3 + [None] * 55,
                       "STAND_TYPE": ["C", "R"] * 59, "X": [1.0, np.inf] * 59})
    ap = pd.DataFrame({"ADES_mvt": ["LTBA", "KJFK", "ZZZZ", None], "STAND_TYPE": ["C", "Q", None, "R"],
                       "X": ["5", "abc", -np.inf, 2.0]})
    X_tr, X_ap = FT.matrices(tr, ap, ["ADES_mvt", "STAND_TYPE", "X"])
    assert list(X_tr["ADES_mvt"].cat.categories) == ["LTBA", "OTHER", "nan"]      # KJFK < 50 → OTHER
    assert list(X_ap["ADES_mvt"]) == ["LTBA", "OTHER", "OTHER", "nan"]
    assert X_ap["STAND_TYPE"].isna().tolist() == [False, True, True, False]         # görülmemiş → NaN
    assert X_tr["X"].isna().sum() == 59 and X_ap["X"].isna().tolist() == [False, True, True, False]
    assert X_tr["X"].dtype == np.float32


if __name__ == "__main__":
    test_matrices()
    print("OK  test_matrices")

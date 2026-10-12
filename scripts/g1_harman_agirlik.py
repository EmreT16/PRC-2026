"""G1 ön bakışı — B1 bileşenleri (LGB 3 tohum, CB, XGB) için harman ağırlıkları.

Ağırlıklar bir bölmede seçilir (0,05 adımlı simpleks ızgarası, alt sınırlı ağırlıklı RMSE), diğer bölmede v126
ağırlıklarına (0,45 CB / 0,40 LGB / 0,15 XGB) karşı 4 kabul kuralıyla sınanır. Seçim ve sınama aynı bölmede
yapılmaz. Eğitim yok.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, models_v126 as M, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
GRID = [(round(float(c), 2), round(float(l), 2), round(float(1 - c - l), 2))
        for c, l in itertools.product(np.arange(0, 1.0001, 0.05), repeat=2) if c + l <= 1.0001]


def load(split):
    d = pd.read_parquet(PRED_DIR / f"B1_{split}.parquet")
    return d, V.month_weights(V.SPLITS[split])


def score(d, w, wts):
    p = wts[0] * d["cb"] + wts[1] * d["lgb"] + wts[2] * d["xgb"]
    return V.weighted_rmse(d["y"], V.floor_offset(p, d["NM_TAXI_DIFF_SEC"]), d["MVT_MONTH"], w)["weighted_rmse"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = {s: load(s) for s in V.SPLITS}
    v126 = (M.WEIGHTS["cb"], M.WEIGHTS["lgb"], M.WEIGHTS["xgb"])
    for pick, test in (("split_1", "split_2"), ("split_2", "split_1")):
        d, w = data[pick]
        scores = {g: score(d, w, g) for g in GRID}
        best = min(scores, key=scores.get)
        print(f"[{pick}'de seçim] en iyi (CB, LGB, XGB) = {best}: {scores[best]:.3f} | v126 ağırlıkları "
              f"{scores[tuple(round(x, 2) for x in v126)]:.3f}")
        dt, wt = data[test]
        p_old = v126[0] * dt["cb"] + v126[1] * dt["lgb"] + v126[2] * dt["xgb"]
        p_new = best[0] * dt["cb"] + best[1] * dt["lgb"] + best[2] * dt["xgb"]
        res = V.compare(dt["y"], p_old, p_new, dt["MVT_MONTH"], V.day_of_year(dt["MVT_MONTH"], dt["MVT_DAY"]),
                        V.SPLITS[test], nm_diff=dt["NM_TAXI_DIFF_SEC"])
        print(V.format_comparison(res, f"{test}'de sınama: {best} ↔ v126 {v126}"))
        V.log_experiment({"ad": f"G1_harman_{pick}_secim_{test}_sinama", "tur": "G1", "secilen": best,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kabul": res["accepted"]}})


if __name__ == "__main__":
    main()

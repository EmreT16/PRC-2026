"""G1 ek — kayıtlı tahminlerle harman analizleri (eğitim yok).

1. Dört bileşen: B1'in LGB (3 tohum), CB, XGB'si + A2'nin global derin CatBoost'u (GCB). Ağırlıklar bir bölmede
   seçilir (0,05 adımlı simpleks), diğer bölmede v126 harmanına (B1 harmanı) karşı sınanır.
2. E1 kısmi harmanı: (1 − β)·lgb_s42 + β·E1_karışım, β bir bölmede seçilir, diğerinde lgb_s42'ye karşı sınanır;
   tüm havalimanları ve yalnızca LIRF.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
STEP = 0.05


def load(split):
    d = pd.read_parquet(PRED_DIR / f"B1_{split}.parquet")
    for f, cols in ((f"A2_global_cb_{split}", ["global_cb"]), (f"E1_{split}", ["karisim"])):
        d = d.merge(pd.read_parquet(PRED_DIR / f"{f}.parquet")[["MVT_ID_mvt"] + cols], on="MVT_ID_mvt",
                    how="left", validate="one_to_one")
    assert d[["global_cb", "karisim"]].notna().all().all()
    d["days"] = V.day_of_year(d["MVT_MONTH"], d["MVT_DAY"])
    return d


def wrmse(d, p, split):
    return V.weighted_rmse(d["y"], V.floor_offset(p, d["NM_TAXI_DIFF_SEC"]), d["MVT_MONTH"],
                           V.month_weights(V.SPLITS[split]))["weighted_rmse"]


def cross(data, name, candidates, make, base_col):
    """candidates: parametre listesi; make(d, param) → tahmin. Seçim bir bölmede, sınama diğerinde."""
    for pick, test in (("split_1", "split_2"), ("split_2", "split_1")):
        d = data[pick]
        sc = {c: wrmse(d, make(d, c), pick) for c in candidates}
        best = min(sc, key=sc.get)
        dt = data[test]
        res = V.compare(dt["y"], dt[base_col], make(dt, best), dt["MVT_MONTH"], dt["days"], V.SPLITS[test],
                        nm_diff=dt["NM_TAXI_DIFF_SEC"])
        print(f"[{name}] {pick}'de seçilen {best} (seçim bölmesinde {sc[best]:.3f})", flush=True)
        print(V.format_comparison(res, f"{test}'de sınama ↔ {base_col}"), flush=True)
        V.log_experiment({"ad": f"G1ek_{name}_{pick}_secim", "tur": "G1", "secilen": best, "sinama": test,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kabul": res["accepted"]}})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = {s: load(s) for s in V.SPLITS}
    g = np.round(np.arange(0, 1 + 1e-9, STEP), 2)
    grid4 = [(float(a), float(b), float(c), round(1 - a - b - c, 2))
             for a, b, c in itertools.product(g, repeat=3) if a + b + c <= 1 + 1e-9]
    cross(data, "dort_bilesen(CB,LGB,XGB,GCB)", grid4,
          lambda d, w: w[0] * d["cb"] + w[1] * d["lgb"] + w[2] * d["xgb"] + w[3] * d["global_cb"], "blend")
    betas = [0.25, 0.5, 0.75, 1.0]
    cross(data, "E1_tum_beta", betas, lambda d, b: (1 - b) * d["lgb_s42"] + b * d["karisim"], "lgb_s42")
    cross(data, "E1_LIRF_beta", betas,
          lambda d, b: np.where(d["ADEP_mvt"].eq("LIRF"), (1 - b) * d["lgb_s42"] + b * d["karisim"], d["lgb_s42"]),
          "lgb_s42")


if __name__ == "__main__":
    main()

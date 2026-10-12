"""C — Özellik kümesi, vekil (hızlı vekil kuralı).

Tek tohumlu LightGBM (v126 ayarları), havalimanı başına. Taban: B1'in lgb_s42'si (v126'nın 134 özelliği).
Adaylar: C123 (temiz küme, TE'li; features_temiz) ve C1234 (temiz küme, TE'siz).
Tahminler data/preds/C_<aday>_<split>.parquet.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_temiz as FT, features_v126 as F, models_v126 as M, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
CANDIDATES = {"C123": True, "C1234_TEsiz": False}


def run(split, candidates=CANDIDATES):
    t0 = time.time()
    train, val, vm = F.prepare_split(split)
    y_tr, y_va = train[config.TARGET].to_numpy(float), val[config.TARGET].to_numpy(float)
    b1 = val[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED_DIR / f"B1_{split}.parquet"), on="MVT_ID_mvt", how="left",
                                   validate="one_to_one")
    assert b1["lgb_s42"].notna().all() and np.allclose(b1["y"], y_va)
    days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
    nm = val["NM_TAXI_DIFF_SEC"].to_numpy(float)
    for name, te in candidates.items():
        tc = time.time()
        feats = FT.feature_list(train.columns, te=te)
        pred = np.full(len(val), np.nan)
        for apt in sorted(val["ADEP_mvt"].unique()):
            tr_i, va_i = (train["ADEP_mvt"] == apt).to_numpy(), (val["ADEP_mvt"] == apt).to_numpy()
            X_tr, X_va = FT.matrices(train[tr_i], val[va_i], feats)
            pred[va_i] = M.lgb_single(X_tr, y_tr[tr_i], X_va)
        res = V.compare(y_va, b1["lgb_s42"], pred, val["MVT_MONTH"], days, vm, nm_diff=nm)
        print(V.format_comparison(res, f"{split}: {name} ({len(feats)} özellik) ↔ lgb_s42 (134)"), flush=True)
        V.log_experiment({"ad": f"{name}_{split}", "tur": "C", "split": split, "taban": "B1 lgb_s42",
                          "ozellik_sayisi": len(feats), "ozellikler": feats,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                    "kabul": res["accepted"]}, "sure_sn": round(time.time() - tc, 1)})
        pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "pred": pred}).to_parquet(
            PRED_DIR / f"C_{name}_{split}.parquet", index=False)
    print(f"[{split}] süre {time.time() - t0:.0f} sn", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for split in ("split_1", "split_2"):
        run(split)


if __name__ == "__main__":
    main()

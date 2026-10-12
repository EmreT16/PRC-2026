"""D2 — Huber kaybı, vekil. Tek tohumlu LightGBM, v126 özellikleri ve ayarları; yalnızca amaç
fonksiyonu değişir (objective=huber, alpha=δ). Taban: B1'in lgb_s42'si (MSE).

Önce split_1'de δ ∈ {300, 600, 1000}; split_1'de kabul alan en iyi δ split_2'de onaylanır.
Kullanım: python scripts/d2_huber.py [δ ...]
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, models_v126 as M, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"


def run(split, deltas):
    t0 = time.time()
    train, val, vm = F.prepare_split(split)
    feats = F.feature_list(train.columns)
    y_tr, y_va = train[config.TARGET].to_numpy(float), val[config.TARGET].to_numpy(float)
    b1 = val[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED_DIR / f"B1_{split}.parquet"), on="MVT_ID_mvt", how="left",
                                   validate="one_to_one")
    assert b1["lgb_s42"].notna().all() and np.allclose(b1["y"], y_va)
    days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
    nm = val["NM_TAXI_DIFF_SEC"].to_numpy(float)
    mats = {}
    for apt in sorted(val["ADEP_mvt"].unique()):
        tr_i, va_i = (train["ADEP_mvt"] == apt).to_numpy(), (val["ADEP_mvt"] == apt).to_numpy()
        mats[apt] = (F.matrix(train[tr_i], feats), y_tr[tr_i], F.matrix(val[va_i], feats), va_i)
    del train
    results = {}
    for d in deltas:
        pred = np.full(len(val), np.nan)
        params = {**M.LGB_PARAMS, "objective": "huber", "alpha": float(d), "random_state": config.SEED}
        for apt, (X_tr, y, X_va, va_i) in mats.items():
            pred[va_i] = lgb.train(params, lgb.Dataset(X_tr, label=y), num_boost_round=M.LGB_ROUNDS).predict(X_va)
        res = V.compare(y_va, b1["lgb_s42"], pred, val["MVT_MONTH"], days, vm, nm_diff=nm)
        print(V.format_comparison(res, f"{split}: Huber δ={d} ↔ MSE (lgb_s42)"), flush=True)
        V.log_experiment({"ad": f"D2_huber_{d}_{split}", "tur": "D2", "split": split, "taban": "B1 lgb_s42",
                          "delta_huber": d,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                    "kabul": res["accepted"]}})
        pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "pred": pred}).to_parquet(
            PRED_DIR / f"D2_huber_{d}_{split}.parquet", index=False)
        results[d] = res
    print(f"[{split}] süre {time.time() - t0:.0f} sn", flush=True)
    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    deltas = [int(x) for x in sys.argv[1:]] or [300, 600, 1000]
    res = run("split_1", deltas)
    ok = {d: r for d, r in res.items() if r["accepted"]}
    if ok:
        best = min(ok, key=lambda d: ok[d]["delta_weighted_rmse"])
        print(f"split_1'de kabul alan en iyi δ = {best}; split_2'de onay", flush=True)
        run("split_2", [best])
    else:
        print("split_1'de hiçbir δ kabul almadı; split_2 çalıştırılmadı", flush=True)


if __name__ == "__main__":
    main()

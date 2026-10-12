"""A1 duman testi: düzeneği gerçek veriyle uçtan uca çalıştırır. Model kararı değildir.

Beklenenler:
- LightGBM, havalimanı medyan ofsetine göre açıkça KABUL almalı (split_1 ve split_2).
- Aynı LightGBM'in farklı tohumlu kopyası RET almalı (sıfır karşılaştırması).
Çıktı: kutuk\\deneyler.jsonl
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, validation as V  # noqa: E402

NUM = ["AOBT_DIFF_SEC", "EOBT_DIFF_SEC", "AOBT_MINUS_EOBT", "TAKEOFF_VS_SCHED_SEC", "LOCAL_HOUR",
       "MVT_DAYOFWEEK", "Q_RWY_DEP_30", "Q_PUSHBACK_30", "DEP_QUEUE_N", "RWY_QUEUE_N",
       "UNIMPEDED_TAXI", "QUEUE_DELAY_EST", "BASE_TAXI_TIME", "TEMPERATURE", "WIND_SPEED"]
CAT = ["ADEP_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]
PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1)


def fit_predict(X, y, tr, va, day_ids, seed):
    """İç gün bloklu erken durdurmayla eğitir; doğrulama tahmini ve tur sayısını döndürür."""
    idx_tr = np.flatnonzero(tr)
    es = V.inner_es_mask(day_ids[idx_tr], seed=seed)
    fit_i, es_i = idx_tr[~es], idx_tr[es]
    params = {**PARAMS, "seed": seed}
    dtr = lgb.Dataset(X.iloc[fit_i], y[fit_i])
    des = lgb.Dataset(X.iloc[es_i], y[es_i], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model.predict(X.iloc[np.flatnonzero(va)], num_iteration=model.best_iteration), model.best_iteration


def summary(res):
    return {"taban_rmse": res["base"]["weighted_rmse"], "aday_rmse": res["new"]["weighted_rmse"],
            "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
            "aylar": {m: [res["base"]["months"][m]["rmse"], res["new"]["months"][m]["rmse"]] for m in res["new"]["months"]},
            "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"], "kabul": res["accepted"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    df = data.load_model_a(NUM + CAT[1:] + ["STAND_mvt"])
    # fillna("nan"): TE v2'nin (pandas < 3) anahtarlarıyla aynı; pandas 3'te astype(str) NaN'ı korur
    df["STAND_RWY"] = (df["ADEP_mvt"].astype(str) + "_"
                       + df["STAND_mvt"].fillna("nan").astype(str).str.strip().str.upper() + "_"
                       + df["RUNWAY_mvt"].fillna("nan").astype(str).str.strip().str.upper())
    for c in CAT:
        df[c] = df[c].astype("category")
    y = df[config.TARGET].to_numpy(float)
    day_ids = df["DAY_ID"].to_numpy()
    print(f"Model A satırı: {len(df):,} | yükleme {time.time() - t0:.0f} sn")

    for split in ("split_1", "split_2"):
        t1 = time.time()
        tr, va, vm = V.split_masks(df["MVT_MONTH"], split)
        med = df.loc[tr].groupby("ADEP_mvt", observed=True)[config.TARGET].median()
        p_base = df.loc[va, "ADEP_mvt"].map(med).astype(float).to_numpy()

        te_tr, te_va = V.grouped_oof_te(df.loc[tr, "STAND_RWY"], y[tr], df.loc[tr, "MVT_MONTH"],
                                        df.loc[va, "STAND_RWY"], m=20.0)
        X = df[NUM + CAT].copy()
        X["STAND_RWY_TE"] = np.float32(np.nan)
        X.loc[tr, "STAND_RWY_TE"] = te_tr
        X.loc[va, "STAND_RWY_TE"] = te_va

        p_new, it = fit_predict(X, y, tr, va, day_ids, seed=config.SEED)
        res = V.compare(y[va], p_base, p_new, df.loc[va, "MVT_MONTH"], day_ids[va], vm,
                        nm_diff=df.loc[va, "NM_TAXI_DIFF_SEC"])
        print(V.format_comparison(res, f"{split}: LightGBM (tur {it}) ↔ havalimanı medyanı"))
        V.log_experiment({"ad": f"A1_duman_{split}", "tur": "duman", "split": split,
                          "taban": "havalimani_medyan_ofset", "aday": "lgb_duman",
                          "ozellikler": NUM + CAT + ["STAND_RWY_TE(ay-OOF, m=20)"],
                          "parametreler": {**PARAMS, "tur": it}, "sonuc": summary(res),
                          "sure_sn": round(time.time() - t1, 1)})

        if split == "split_1":
            p_seed, it2 = fit_predict(X, y, tr, va, day_ids, seed=config.SEED + 1)
            res0 = V.compare(y[va], p_new, p_seed, df.loc[va, "MVT_MONTH"], day_ids[va], vm,
                             nm_diff=df.loc[va, "NM_TAXI_DIFF_SEC"])
            print(V.format_comparison(res0, f"{split}: sıfır karşılaştırması (aynı model, tohum +1, tur {it2})"))
            V.log_experiment({"ad": "A1_sifir_split_1", "tur": "duman", "split": split,
                              "taban": "lgb_duman", "aday": "lgb_duman_tohum+1", "sonuc": summary(res0)})
        print(f"  süre {time.time() - t1:.0f} sn")
    print(f"toplam {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()

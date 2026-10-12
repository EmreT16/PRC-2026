"""A2 — Düzenek kalibrasyonu, benzer deneylerle. B1 tahminlerini gerektirir.

1. Eğitim içi TE: tek tohumlu LightGBM (v126 parametreleri); eğitim satırlarının TE'si kendi hedefi dahil.
   Karşılaştırma: B1'deki aynı ayarlı tek tohumlu LightGBM (lgb_s42, kat içi TE). Beklenen: kötüleşme
   (v21 → v26: LB +10,46).
2. Global derin CatBoost %50: tüm havalimanlarında tek CatBoost (depth 9, GPU; hedef ofset ±7.200'de kırpık;
   ADEP + 5 kategorik), B1 harmanıyla %50. Beklenen: iyileşme (v21 → v44: LB −1,26).
   Süre için lr 0,06 ve en fazla 3.000 iterasyon (özgün: lr 0,03, 5.000); erken durdurma eğitim günlerinden iç kümede.
Uyarı: yönler eski tabanlarda ölçüldü; yeni tabanda farklı çıkabilir. Uyuşmazlık düzeneği çürütmez.
Tüm karşılaştırmalar gönderimdeki 60 sn alt sınırla (validation.floor_offset). Tahminler data/preds/A2_*.parquet.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, models_v126 as M, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
SPLITS = ("split_1", "split_2")


def b1_aligned(split, val):
    b1 = pd.read_parquet(PRED_DIR / f"B1_{split}.parquet")
    b1 = val[["MVT_ID_mvt"]].merge(b1, on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert b1["blend"].notna().all()
    return b1


def report(name, split, res, extra=None):
    print(V.format_comparison(res, f"{split}: {name}"), flush=True)
    V.log_experiment({"ad": f"A2_{name}_{split}", "tur": "A2", "split": split,
                      "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                "kabul": res["accepted"]}, **(extra or {})})


def insample_te(split):
    t0 = time.time()
    train, val, vm = F.prepare_split(split, insample_te=True)
    feats = F.feature_list(train.columns)
    pred = np.full(len(val), np.nan)
    for apt in sorted(val["ADEP_mvt"].unique()):
        tr_i, va_i = (train["ADEP_mvt"] == apt).to_numpy(), (val["ADEP_mvt"] == apt).to_numpy()
        pred[va_i] = M.lgb_single(F.matrix(train[tr_i], feats), train.loc[tr_i, config.TARGET].to_numpy(float),
                                  F.matrix(val[va_i], feats))
    b1 = b1_aligned(split, val)
    days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
    res = V.compare(b1["y"], b1["lgb_s42"], pred, val["MVT_MONTH"], days, vm, nm_diff=b1["NM_TAXI_DIFF_SEC"])
    report("egitim_ici_TE", split, res, {"beklenen": "kötüleşme", "sure_sn": round(time.time() - t0, 1)})
    pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "lgb_s42_egitim_ici_te": pred}).to_parquet(
        PRED_DIR / f"A2_egitim_ici_te_{split}.parquet", index=False)


def global_catboost(split):
    t0 = time.time()
    train, val, vm = F.prepare_split(split)
    feats = F.feature_list(train.columns)
    cats = ["ADEP_mvt"] + F.CAT_V126

    def frame(df):
        n, num = F.numeric_array(F.matrix(df, feats), feats)
        return pd.concat([pd.DataFrame(n, columns=num), df[cats].fillna("nan").astype(str).reset_index(drop=True)],
                         axis=1)

    d_tr, d_va = frame(train), frame(val)
    y = np.clip(train[config.TARGET].to_numpy(float), -7200, 7200)
    es = V.inner_es_mask(V.day_of_year(train["MVT_MONTH"], train["MVT_DAY"]))
    params = dict(iterations=3000, learning_rate=0.06, depth=9, task_type="GPU", random_seed=42, verbose=500,
                  od_type="Iter", od_wait=200, use_best_model=True)
    try:
        model = CatBoostRegressor(**params).fit(Pool(d_tr[~es], y[~es], cat_features=cats),
                                                eval_set=Pool(d_tr[es], y[es], cat_features=cats))
    except Exception as e:  # noqa: BLE001 — raporlanır; bilinçli ve kayıtlı yeniden deneme
        print(f"  CatBoost HATA (varsayılan max_ctr_complexity): {type(e).__name__}: {str(e)[:200]}", flush=True)
        params["max_ctr_complexity"] = 1
        model = CatBoostRegressor(**params).fit(Pool(d_tr[~es], y[~es], cat_features=cats),
                                                eval_set=Pool(d_tr[es], y[es], cat_features=cats))
    p_glob = model.predict(Pool(d_va, cat_features=cats))
    b1 = b1_aligned(split, val)
    days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
    extra = {"iterasyon": model.get_best_iteration(), "parametreler": params, "sure_sn": round(time.time() - t0, 1)}
    nm = b1["NM_TAXI_DIFF_SEC"]
    report("global_CB_tek_basina", split,
           V.compare(b1["y"], b1["blend"], p_glob, val["MVT_MONTH"], days, vm, nm_diff=nm), extra)
    report("global_CB_yuzde50", split,
           V.compare(b1["y"], b1["blend"], 0.5 * b1["blend"] + 0.5 * p_glob, val["MVT_MONTH"], days, vm, nm_diff=nm),
           {**extra, "beklenen": "iyileşme"})
    pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "global_cb": p_glob}).to_parquet(
        PRED_DIR / f"A2_global_cb_{split}.parquet", index=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for split in SPLITS:
        insample_te(split)
    for split in SPLITS:
        global_catboost(split)


if __name__ == "__main__":
    main()

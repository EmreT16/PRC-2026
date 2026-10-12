"""F3 — ARVT_1 türevleri, vekil. Tek tohumlu LightGBM; v126'nın 134 özelliği + 3.

Ham zamanlardan (PRC\\data\\training_2025-*.parquet ve ranking.parquet; yalnızca 6 sütun, salt okunur):
- F3_PLAN_SURE   = ARVT_1 − EOBT_1         (ilk plandaki kapıdan varışa süre)
- F3_ARVT_SAPMA  = ARVT_3 − ARVT_1         (varış tahmininin kayması)
- F3_PLAN_TAKSI  = (ARVT_1 − EOBT_1) − (ARVT_3 − MVT)   (plan süresi − gerçekleşen uçuş süresi ≈ planlı taksi + uçuş
                                                          süresi farkı)
Özellik tabloları data/f3_arvt_<train|rank>.parquet. Taban: B1'in lgb_s42'si.
"""
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, models_v126 as M, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
RAW_DIR = config.PRC2_DIR / "data"
COLS = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "EOBT_1_flt", "ARVT_1_flt", "ARVT_3_flt"]
F3_COLS = ["F3_PLAN_SURE", "F3_ARVT_SAPMA", "F3_PLAN_TAKSI"]


def build(which):
    path = config.DATA_DIR / f"f3_arvt_{which}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    files = sorted(glob.glob(str(RAW_DIR / "training_2025-*.parquet"))) if which == "train" \
        else [str(config.RANK_RAW)]
    raw = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    sec = lambda a, b: (raw[a] - raw[b]).dt.total_seconds().astype(np.float32)  # noqa: E731
    out = pd.DataFrame({"MVT_ID_mvt": raw["MVT_ID_mvt"].to_numpy(),
                        "F3_PLAN_SURE": sec("ARVT_1_flt", "EOBT_1_flt"),
                        "F3_ARVT_SAPMA": sec("ARVT_3_flt", "ARVT_1_flt")})
    out["F3_PLAN_TAKSI"] = out["F3_PLAN_SURE"] - sec("ARVT_3_flt", "MVT_TIME_UTC_mvt")
    assert out["MVT_ID_mvt"].is_unique
    out.to_parquet(path, index=False)
    return out


def run(split, f3):
    t0 = time.time()
    train, val, vm = F.prepare_split(split)
    train = train.merge(f3, on="MVT_ID_mvt", how="left", validate="one_to_one")
    val = val.merge(f3, on="MVT_ID_mvt", how="left", validate="one_to_one")
    print(f"[{split}] F3 NaN: eğitim {train[F3_COLS].isna().mean().round(4).to_dict()} | doğrulama "
          f"{val[F3_COLS].isna().mean().round(4).to_dict()}", flush=True)
    feats = F.feature_list([c for c in train.columns if c not in F3_COLS]) + F3_COLS
    y_tr, y_va = train[config.TARGET].to_numpy(float), val[config.TARGET].to_numpy(float)
    pred = np.full(len(val), np.nan)
    for apt in sorted(val["ADEP_mvt"].unique()):
        tr_i, va_i = (train["ADEP_mvt"] == apt).to_numpy(), (val["ADEP_mvt"] == apt).to_numpy()
        pred[va_i] = M.lgb_single(F.matrix(train[tr_i], feats), y_tr[tr_i], F.matrix(val[va_i], feats))
    b1 = val[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED_DIR / f"B1_{split}.parquet"), on="MVT_ID_mvt", how="left",
                                   validate="one_to_one")
    assert b1["lgb_s42"].notna().all() and np.allclose(b1["y"], y_va)
    res = V.compare(y_va, b1["lgb_s42"], pred, val["MVT_MONTH"], V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"]), vm,
                    nm_diff=val["NM_TAXI_DIFF_SEC"].to_numpy(float))
    print(V.format_comparison(res, f"{split}: F3 ({len(feats)} özellik) ↔ lgb_s42 (134)"), flush=True)
    V.log_experiment({"ad": f"F3_arvt_{split}", "tur": "F", "split": split, "taban": "B1 lgb_s42",
                      "eklenen": F3_COLS,
                      "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                "kabul": res["accepted"]}, "sure_sn": round(time.time() - t0, 1)})
    pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "pred": pred}).to_parquet(
        PRED_DIR / f"F3_{split}.parquet", index=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    f3 = build("train")
    build("rank")
    for split in ("split_1", "split_2"):
        run(split, f3)


if __name__ == "__main__":
    main()

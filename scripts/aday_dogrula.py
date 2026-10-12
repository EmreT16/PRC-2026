"""Adayı düzenekte tri-modelle doğrular (split_1 ve split_2), B1 harmanına (v126 YENİ_200) karşı.

Kullanım: python scripts/aday_dogrula.py <aday> [split_1|split_2]
Aday tanımları src/aday.py (ADAYLAR). Tahminler data/preds/ADAY_<aday>_<split>.parquet.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import aday as AD, config, features_v126 as F, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"


def run(name, split):
    spec = AD.ADAYLAR[name]
    t0 = time.time()
    assert not F.KUYRUK or "_kq" in name, "PRC_KUYRUK=1 iken aday adı '_kq' içermeli"
    assert F.GECIKME == ("_kqg" in name), "PRC_GECIKME=1 ile aday adındaki '_kqg' uyuşmalı"
    assert F.GECPIST == ("_kqgp" in name), "PRC_GECPIST=1 ile aday adındaki '_kqgp' uyuşmalı"
    assert F.EKOZ == ("_kqgpy" in name), "PRC_EKOZ=1 ile aday adındaki '_kqgpy' uyuşmalı"
    train, val, vm = F.prepare_split(split)
    train, val = AD.kuyruk_ekle(spec, train, val)
    feats = AD.features(spec, train.columns)
    y_va = val[config.TARGET].to_numpy(float)
    preds = {}
    for apt in sorted(val["ADEP_mvt"].unique()):
        ta = time.time()
        tr_i, va_i = (train["ADEP_mvt"] == apt).to_numpy(), (val["ADEP_mvt"] == apt).to_numpy()
        out = AD.fit_predict_airport(spec, train[tr_i], val[va_i], feats)
        for k, v in out.items():
            preds.setdefault(k, np.full(len(val), np.nan))[va_i] = v
        print(f"  {apt}: harman {np.sqrt(np.mean((out['harman'] - y_va[va_i]) ** 2)):.1f} (alt sınırsız) | "
              f"{time.time() - ta:.0f} sn", flush=True)
    b1 = val[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED_DIR / f"B1_{split}.parquet"), on="MVT_ID_mvt", how="left",
                                   validate="one_to_one")
    assert b1["blend"].notna().all() and np.allclose(b1["y"], y_va)
    days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
    res = V.compare(y_va, b1["blend"], preds["harman"], val["MVT_MONTH"], days, vm,
                    nm_diff=val["NM_TAXI_DIFF_SEC"].to_numpy(float))
    print(V.format_comparison(res, f"{split}: {name} ({len(feats)} özellik) ↔ B1 harmanı"), flush=True)
    V.log_experiment({"ad": f"ADAY_{name}_{split}", "tur": "aday", "split": split, "taban": "B1 harman",
                      "tanim": spec, "ozellik_sayisi": len(feats),
                      "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                "kabul": res["accepted"]}, "sure_sn": round(time.time() - t0, 1)})
    pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), **preds}).to_parquet(
        PRED_DIR / f"ADAY_{name}_{split}.parquet", index=False)
    print(f"[{split}] süre {time.time() - t0:.0f} sn", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for s in (sys.argv[2:] or ["split_1", "split_2"]):
        run(sys.argv[1], s)

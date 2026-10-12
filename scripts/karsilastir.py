"""İki tahmin kümesinin doğrudan karşılaştırması, iki bölmede (eğitim yok; 4 kabul kuralı, 60 sn alt sınırlı).

Kullanım: python scripts/karsilastir.py <taban> <aday>
<taban>/<aday>: B1 sütunu (cb, lgb, xgb, lgb_s42, blend) ya da <dosya>:<sütun> (data/preds/<dosya>_<split>.parquet).
  ör. python scripts/karsilastir.py A2_global_cb:global_cb GCB_kat_ek5:global_cb
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"


def column(d, spec, split):
    if ":" not in spec:
        return d[spec]
    f, col = spec.split(":", 1)
    b = pd.read_parquet(PRED_DIR / f"{f}_{split}.parquet", columns=["MVT_ID_mvt", col])
    return d[["MVT_ID_mvt"]].merge(b, on="MVT_ID_mvt", how="left", validate="one_to_one")[col]


def main(base, new):
    for s in V.SPLITS:
        d = pd.read_parquet(PRED_DIR / f"B1_{s}.parquet")
        pb, pn = column(d, base, s), column(d, new, s)
        assert pb.notna().all() and pn.notna().all()
        res = V.compare(d["y"], pb, pn, d["MVT_MONTH"], V.day_of_year(d["MVT_MONTH"], d["MVT_DAY"]), V.SPLITS[s],
                        nm_diff=d["NM_TAXI_DIFF_SEC"])
        print(V.format_comparison(res, f"{s}: {new} ↔ {base}"), flush=True)
        V.log_experiment({"ad": f"KARSILASTIR_{new}_vs_{base}_{s}", "tur": "karşılaştırma", "split": s,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                    "kabul": res["accepted"]}})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2])

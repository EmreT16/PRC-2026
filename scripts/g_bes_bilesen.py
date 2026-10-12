"""G1 beş bileşenli harmanın test tahmini (ağırlıklar kutuk/g1_agirliklar.json).

Bileşenler: TEST_tam_lgb_xgb (lgb, xgb), TEST_gcb (global_cb), TEST_tam_e1_lgb42 (harman = E1 karışımı).
Havalimanı bazlı CB'nin ağırlığı 0 olmalı (değilse hata). Çıktı: data/preds/TEST_bes_bilesen.parquet (taksi dahil).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

PRED = config.DATA_DIR / "preds"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g = json.loads((config.LOG_DIR / "g1_agirliklar.json").read_text(encoding="utf-8"))
    w = dict(zip(g["bilesenler"], g["agirlik"]))
    assert w["cb"] == 0, w
    a = pd.read_parquet(PRED / "TEST_tam_lgb_xgb.parquet")[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "lgb", "xgb"]]
    for f, src, dst in (("TEST_gcb", "global_cb", "global_cb"), ("TEST_tam_e1_lgb42", "harman", "karisim")):
        b = pd.read_parquet(PRED / f"{f}.parquet")[["MVT_ID_mvt", src]].rename(columns={src: dst})
        a = a.merge(b, on="MVT_ID_mvt", how="left", validate="one_to_one")
    comps = [c for c in g["bilesenler"] if w[c] > 0]
    assert a[comps].notna().all().all() and len(a) == 339_551
    a["harman"] = sum(w[c] * a[c] for c in comps) / sum(w[c] for c in comps)
    a["taksi"] = np.maximum(V.TAXI_FLOOR, a["NM_TAXI_DIFF_SEC"] - a["harman"])
    a.to_parquet(PRED / "TEST_bes_bilesen.parquet", index=False)
    print(f"ağırlıklar {w} | ort. taksi {a['taksi'].mean():.1f} | bileşen korelasyonları:\n"
          f"{a[comps].corr().round(3).to_string()}")


if __name__ == "__main__":
    main()

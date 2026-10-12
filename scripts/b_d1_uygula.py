"""NM'siz satırlarda LTFM / LFPG D1 kalıntı düzeltmesini
bir B tahmin dosyasına uygular. Fark (d1 − mevcut) extra_data/b_deney2_test.parquet'ten
(v190'ın B tahmini üstüne hesaplandı); LTFM'de B tahmini sürümler arasında aynı, LFPG'de ADS-B düzeltmesi yüzünden en çok
~±60 sn farklı, bu yüzden farkı yeniden kullanmak yaklaşık ama yeterli.
Kullanım: python scripts/b_d1_uygula.py <B_ad>   → data/preds/TEST_B_<B_ad>_d1.parquet
05.10 (YEDEK Y2b, eğitime dayalı kural): python scripts/b_d1_uygula.py <B_ad> LTFM=1 LFPG=0 → TEST_B_<B_ad>_d1_LTFM1_LFPG0
(havalimanı başına ölçek k; LFPG'de 2026 düzeltmesi 2025'in dışına taşıyor (ort. +7 → +108 sn) → k = 0).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

FARK = config.EK_VERI / "b_deney2_test.parquet"
APT = ("LTFM", "LFPG")


def main(ad, kmap=None):
    sys.stdout.reconfigure(encoding="utf-8")
    kmap = kmap or {a: 1.0 for a in APT}
    b = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_B_{ad}.parquet")
    r = pd.read_parquet(FARK)
    r["fark"] = (r["d1"] - r["mevcut"]) * r["ADEP_mvt"].map(kmap).fillna(0.0)
    b = b.merge(r[["MVT_ID_mvt", "fark"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert b["fark"].notna().all(), "B satırlarının hepsinde fark olmalı"
    b["taksi"] = np.maximum(60.0, b["taksi"] + b["fark"])
    print(f"{ad}: {int((b['fark'] != 0).sum())} satır değişti; ort. {b.loc[b['fark'] != 0, 'fark'].mean():+.1f} sn")
    ek = "_d1" if kmap == {a: 1.0 for a in APT} else "_d1_" + "_".join(f"{a}{v:g}" for a, v in kmap.items())
    b.drop(columns="fark").to_parquet(config.DATA_DIR / "preds" / f"TEST_B_{ad}{ek}.parquet", index=False)
    print("çıktı", f"TEST_B_{ad}{ek}.parquet")


if __name__ == "__main__":
    km = {a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[2:] if "=" in a}
    main(sys.argv[1], km or None)

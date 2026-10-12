"""v126 temel tablolarını üretir (data\\base_v126_*_v1.parquet) ve yeniden kurulumu doğrular.

Doğrulamalar:
1. Özellik listesi, şemadan yeniden kurulan v126 listesiyle (134 sütun) birebir aynı mı.
2. Yeniden yazılan TE, test satırlarında PRC_4'ün TE v2 / V3 dosyalarındaki değerleri veriyor mu
   (orada da test TE'si 12 ayın tamamından hesaplanmıştı).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F  # noqa: E402


def reconstructed_v126_list():
    """v126 özellik listesinin yeniden kurulumu (yalnızca şema)."""
    cols = pq.ParquetFile(config.DEP_TRAIN).schema_arrow.names
    for p in (F.PRC4_FEAT / "target_enc_train_v2.parquet", F.V3_FILES["train"]):
        cols += [c for c in pq.ParquetFile(p).schema_arrow.names if c != "MVT_ID_mvt" and c not in cols]
    cols += ["STAND_TO_RWY_DIST_M", "STAND_MIN_DIST_M", "STAND_DIST_RATIO", "AOBT_EOBT_QUEUE_PROD",
             "AOBT_EOBT_QUEUE15_PROD", "QUEUE_DELAY_AOBT_SUM", "CDM_REVISION_MAGNITUDE", "LOBT_IOBT_DIFF",
             "EOBT_AOBT_DIFF", "PEAK_AOBT_EOBT", "AOBT_EOBT_ABS", "QUEUE_CDM_RATIO", "LIRF_NOISY_STAND",
             "LIRF_NOISY_16R", "DEP_COUNT_10", "DEP_COUNT_30", "DEP_COUNT_60", "DEP_RWY_COUNT_10", "DEP_RWY_COUNT_30"]
    return [c for c in cols if c not in F.EXCLUDE_V126 and not c.startswith("DEST_") and "holiday" not in c.lower()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = F.base_table("train")
    rk = F.base_table("rank")
    print(f"temel tablolar: eğitim {tr.shape}, test {rk.shape} | {time.time() - t0:.0f} sn")

    feats = F.feature_list(tr.columns)
    ref = reconstructed_v126_list()
    print(f"özellik sayısı: {len(feats)} | şemadan kurulan liste: {len(ref)} | "
          f"aynı küme: {set(feats) == set(ref)} | aynı sıra: {feats == ref}")
    if set(feats) != set(ref):
        print("  fazla:", sorted(set(feats) - set(ref)), "| eksik:", sorted(set(ref) - set(feats)))

    # TE yeniden kurulumu ↔ TE v2 / V3 test dosyaları (TE v2 temizlenmemiş Model A satırlarıyla hesaplanmıştı)
    a_tr = tr[tr[config.TARGET].notna()].copy()
    a_rk = rk[rk["NM_TAXI_DIFF_SEC"].notna()].copy()
    F.add_te(a_tr, a_rk, a_tr["MVT_MONTH"].to_numpy())
    ref_rk = (pd.read_parquet(F.PRC4_FEAT / "target_enc_rank_v2.parquet")
              .merge(pd.read_parquet(F.V3_FILES["rank"], columns=["MVT_ID_mvt", F.V3_TE_COL]), on="MVT_ID_mvt"))
    cmp = a_rk[["MVT_ID_mvt"] + F.TE_COLS].merge(ref_rk, on="MVT_ID_mvt", suffixes=("", "_ref"))
    print(f"\nTE karşılaştırması (test Model A satırları, n={len(cmp):,}):")
    for c in F.TE_COLS:
        d = (cmp[c] - cmp[f"{c}_ref"]).abs()
        print(f"  {c:22s} maks |fark| {d.max():9.4f} | %99,9 {d.quantile(0.999):8.4f} | "
              f"fark > 0,01 olan satır {(d > 0.01).mean():6.2%} | r {np.corrcoef(cmp[c], cmp[c + '_ref'])[0, 1]:.6f}")
    print(f"toplam {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()

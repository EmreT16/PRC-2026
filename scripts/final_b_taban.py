"""Final aşaması: Model B'nin "eski tabanı" (TEST_B_tum6M_adsb) final testi için (08.10).

Normal testte TEST_B_tum6M_adsb.parquet'in kullanılan kısımları (08.10 ölçümü):
  - EHAM / LSZH / EDDF'in taban sütunu = v126'nın Model B'si (B5 düzelticisi bunun üstüne kurulur; eşleşmeyen satırda taban kalır);
  - LIRF ön koşullu satırların (model_b.on_kosul_24s) taksisi = v126'nın Model B'si (b_lirf_te.py "eski");
  - 6 havalimanı (EDDM, EGLL, LEBL, LEMD, LFPG, LTFM) ve LIRF'in ön koşul dışı satırları sonraki adımlarda yeni modelle değişir;
    eski ilk düzelticinin (adsb_b_duzeltici.py) rhat'ı hiçbir yerde kalmıyor.
v126'nın Model B'si bu depoda model_b.fit_predict_v126b olarak yeniden yazılmıştı; normal testin 5.290 B satırında v126
gönderimini birebir üretiyor (08.10: 10 havalimanının hepsinde fark 0,0). Final testinde bu işlev tüm B satırlarına uygulanır:
taban = taksi = v126 B, rhat = 0.
Kullanım: PRC_FINAL=1 python scripts/final_b_taban.py   → data_final/preds/TEST_B_tum6M_adsb.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    assert config.FINAL, "PRC_FINAL=1 gerekli"
    tr, ap = MB.load_b("train"), MB.load_b("rank")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    o = MB.fit_predict_v126b(tr, ap, feats)
    out = pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": ap["ADEP_mvt"].to_numpy(),
                        "taban": o["taksi"], "taksi": o["taksi"], "rhat": 0.0})
    assert out["taban"].notna().all() and len(out) == config.TEST_N - config.TEST_A_N
    (config.DATA_DIR / "preds").mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.DATA_DIR / "preds" / "TEST_B_tum6M_adsb.parquet", index=False)
    ay = ap["MVT_MONTH"].to_numpy()
    print(f"final B satırı {len(out):,} | ön koşullu {int(MB.on_kosul_24s(ap).sum())}")
    print(pd.crosstab(out["ADEP_mvt"], ay).to_string())
    v126 = config.PRC4_DIR / "submissions" / "strong-moose_v126.parquet"
    if v126.exists():  # 12.10: yalnız tanı amaçlı karşılaştırma; dosya yoksa (temiz kopya) atlanır, çıktı aynı
        x = out.merge(pd.read_parquet(v126), on="MVT_ID_mvt")
        f = x.taban - x.TAXITIME_SEC_mvt
        print(f"Oca / Tem ortak {len(x):,} satır: final v126 B − v126 gönderimi RMS {np.sqrt((f ** 2).mean()):.1f}, "
              f"maks |fark| {f.abs().max():.0f} (fark yalnız ±12 sn oynatma ve 4 aylık grup istatistiklerinden)")


if __name__ == "__main__":
    main()

"""11.10: düzeltici üretimini 3 → 5 tohuma çıkarır, yeniden eğitmeden.
<3 tohumlu dosya> (tohum 42, 2026, 7) ve <2 tohumlu dosya> (aynı ayarlar, tohumlar=1,2) → r̂₅ = 0,6 · r̂₃ + 0,4 · r̂₂; aile ağırlığı
(nn_w) iki dosyada da aynı ve doğrusal olduğu için tek koşu 5 tohumla birebir aynı. taksi = max(60, taban + r̂₅), LIRF yankı
hedge'i eski E1 ile (girdilerle aynı; scripts/hedge_e1_degistir.hedge) → çıktı <3 tohumlu dosya> ile aynı biçimde;
T1 sonra hedge_e1_degistir.py ile.
Kullanım: python scripts/duz_tohum_birlestir.py D_kqgpPeT2_adsb_42p D_kqgpPeT2_adsb_42pq D_kqgpPeT2_adsb_42p5   (PRC_FINAL=1 → data_final)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import adsb as A  # noqa: E402
import hedge_e1_degistir as H  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a3, a2, cik = sys.argv[1:4]
    t3, t2 = pd.read_parquet(A.PRED / f"TEST_{a3}.parquet"), pd.read_parquet(A.PRED / f"TEST_{a2}.parquet")
    assert (t3.MVT_ID_mvt.to_numpy() == t2.MVT_ID_mvt.to_numpy()).all()
    assert np.array_equal(t3.taksi_taban.to_numpy(float), t2.taksi_taban.to_numpy(float))
    r3, r2 = t3.rhat.to_numpy(float), t2.rhat.to_numpy(float)
    assert np.array_equal(r3 != 0, r2 != 0), "izli satır kümeleri farklı"
    m = r3 != 0
    for ad, t in ((a3, t3), (a2, t2)):
        f = np.abs(H.hedge(t, "v_e1_huber1000_3t_kqgp")[0] - t.taksi.to_numpy(float)).max()
        assert f < 1e-6, f"{ad}: eski E1 ile yeniden üretilemedi ({f})"
    print(f"izli {m.sum():,} satır | r̂₃ ↔ r̂₂ korelasyonu {np.corrcoef(r3[m], r2[m])[0, 1]:.4f}, fark RMS "
          f"{np.sqrt(np.mean((r3[m] - r2[m]) ** 2)):.2f}")
    t = t3.copy()
    t["rhat"] = 0.6 * r3 + 0.4 * r2
    t["taksi"] = H.hedge(t, "v_e1_huber1000_3t_kqgp")[0]
    d = t.taksi.to_numpy(float) - t3.taksi.to_numpy(float)
    print(f"5 tohum − 3 tohum: RMS {np.sqrt(np.mean(d[m] ** 2)):.2f} (izli), ort {d[m].mean():+.3f}, maks |fark| {np.abs(d).max():.0f}")
    t.to_parquet(A.PRED / f"TEST_{cik}.parquet", index=False)
    print("yazıldı:", A.PRED / f"TEST_{cik}.parquet")


if __name__ == "__main__":
    main()

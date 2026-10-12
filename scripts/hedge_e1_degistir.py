"""10.10 (T1): LIRF yankı hedge'inde p_kilit kaynağını değiştirir — düzeltici
yeniden eğitilmeden (hedge, düzelticiden sonra uygulanan kapalı bir formül; adsb_duzeltici.uretim ile aynı hesap):
  p = max(60, taban + k · r̂);  LIRF izli satırlarda taksi = max(60, p + β · p_kilit · (max(60, MVT − tarife) − p)), β = 0,25.
Üretimde p_kilit harmanın E1_AD'sinden (kqgpPeT2 → eski kqgp E1, 3 tohum); final E1 = 6 tohum, pist + ekoz'lu (kqgpy).
Önce eski E1 ile mevcut dosyayı birebir üretir (denetim), sonra yeni E1 ile yazar.
Kullanım: python scripts/hedge_e1_degistir.py <girdi adı> <çıktı adı> [e1=v_e1_huber1000_6t_pist_kqgpy] [eski=v_e1_huber1000_3t_kqgp]
  ör. python scripts/hedge_e1_degistir.py D_kqgpPeT2_adsb_42n D_kqgpPeT2_adsb_42nT   (PRC_FINAL=1 → data_final)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

BETA, K = 0.25, 1.0


def hedge(t, e1):
    pk = t[["MVT_ID_mvt"]].merge(pd.read_parquet(A.PRED / f"TEST_{e1}.parquet", columns=["MVT_ID_mvt", "p_kilit"]),
                                 on="MVT_ID_mvt", how="left", validate="one_to_one")["p_kilit"].fillna(0).to_numpy()
    ham = pd.read_parquet(config.RANK_RAW, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    ham["gec"] = (ham["MVT_TIME_UTC_mvt"] - ham["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    g = np.maximum(60.0, t[["MVT_ID_mvt"]].merge(ham[["MVT_ID_mvt", "gec"]], on="MVT_ID_mvt", how="left",
                                                 validate="one_to_one")["gec"].fillna(0).to_numpy(float))
    m = (t["rhat"].abs().to_numpy() > 0) & (t["ADEP_mvt"] == "LIRF").to_numpy()
    p = np.maximum(60.0, t["taksi_taban"].to_numpy(float) + K * t["rhat"].to_numpy(float))
    return np.where(m, np.maximum(60.0, p + BETA * pk * (g - p)), p), m, pk


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    poz = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    t = pd.read_parquet(A.PRED / f"TEST_{poz[0]}.parquet")
    eski, m, pk0 = hedge(t, opt.get("eski", "v_e1_huber1000_3t_kqgp"))
    f0 = np.abs(eski - t["taksi"].to_numpy(float)).max()
    print(f"denetim: eski E1 ile yeniden hesap − mevcut taksi maks |fark| {f0:.6f} sn ({int(m.sum()):,} hedge satırı)")
    assert f0 < 1e-6, "mevcut dosya yeniden üretilemedi"
    yeni, _, pk1 = hedge(t, opt.get("e1", "v_e1_huber1000_6t_pist_kqgpy"))
    d = yeni - eski
    print(f"p_kilit (hedge satırları): eski ort {pk0[m].mean():.4f} → yeni {pk1[m].mean():.4f}, korelasyon "
          f"{np.corrcoef(pk0[m], pk1[m])[0, 1]:.4f} | taksi farkı RMS {np.sqrt(np.mean(d[m] ** 2)):.2f}, ort {d[m].mean():+.2f}, "
          f"maks |fark| {np.abs(d).max():.0f}")
    t["taksi"] = yeni
    t.to_parquet(A.PRED / f"TEST_{poz[1]}.parquet", index=False)
    print("yazıldı:", A.PRED / f"TEST_{poz[1]}.parquet")


if __name__ == "__main__":
    main()

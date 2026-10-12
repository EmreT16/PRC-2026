"""11.10: düzelticinin TabM ağırlığını yeniden eğitmeden değiştirir. Girdi dosyası
`bilesen=1` ile üretilmiş olmalı (rhat_lgb / rhat_nn: aile ortalamaları). r̂ = (1 − v) · rhat_lgb + v · rhat_nn; taksi = max(60,
taban + r̂), LIRF yankı hedge'i eski E1 ile (girdiyle aynı; scripts/hedge_e1_degistir.hedge) → T1 sonra hedge_e1_degistir.py ile.
Önce girdinin kendi v'siyle birebir yeniden üretildiği denetlenir.
Kullanım: python scripts/duz_v_degistir.py <girdi> <çıktı> v=<yeni> v0=<girdinin v'si>   (PRC_FINAL=1 → data_final)
  ör. python scripts/duz_v_degistir.py D_kqgpPeT2_adsb_42p9 D_kqgpPeT2_adsb_42pv v=1.0 v0=0.9
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
    poz = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    v, v0 = float(opt["v"]), float(opt["v0"])
    t = pd.read_parquet(A.PRED / f"TEST_{poz[0]}.parquet")
    L, N = t.rhat_lgb.to_numpy(float), t.rhat_nn.to_numpy(float)
    f0 = np.abs((1 - v0) * L + v0 * N - t.rhat.to_numpy(float)).max()
    e0 = np.abs(H.hedge(t, "v_e1_huber1000_3t_kqgp")[0] - t.taksi.to_numpy(float)).max()
    print(f"denetim: v0 = {v0} ile r̂ maks |fark| {f0:.2e}, taksi (eski E1 hedge) maks |fark| {e0:.2e}")
    assert f0 < 1e-3 and e0 < 1e-6, "girdi yeniden üretilemedi"
    eski = t.taksi.to_numpy(float).copy()
    t["rhat"] = (1 - v) * L + v * N
    t["taksi"] = H.hedge(t, "v_e1_huber1000_3t_kqgp")[0]
    d = t.taksi.to_numpy(float) - eski
    iz = np.abs(L) + np.abs(N) > 0
    print(f"v {v0} → {v}: izli {iz.sum():,} satır, taksi farkı RMS {np.sqrt(np.mean(d[iz] ** 2)):.2f}, ort {d[iz].mean():+.3f}, "
          f"maks |fark| {np.abs(d).max():.0f}")
    t.to_parquet(A.PRED / f"TEST_{poz[1]}.parquet", index=False)
    print("yazıldı:", A.PRED / f"TEST_{poz[1]}.parquet")


if __name__ == "__main__":
    main()

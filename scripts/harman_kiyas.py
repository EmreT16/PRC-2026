"""İki taban harmanının 6 ay OOF taksi RMSE'si (D_dw ağırlıkları, src/adsb.py d_dw_split). Kullanım: harman_kiyas.py <eski> <yeni>"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def tablo(h):
    os.environ["PRC_HARMAN"] = h
    import adsb
    A = importlib.reload(adsb)
    out = {}
    for s in ("split_1", "split_2", "split_3"):
        d = A.d_dw_split(s)
        g = d["NM_TAXI_DIFF_SEC"] - d["y"]
        p = np.maximum(60, d["NM_TAXI_DIFF_SEC"] - d["ofset_taban"])
        for m, i in d.groupby("MVT_MONTH").indices.items():
            out[m] = float(np.sqrt(np.mean((g.to_numpy()[i] - p.to_numpy()[i]) ** 2)))
        for a, i in d.groupby("ADEP_mvt").indices.items():
            out.setdefault(a, []).append(float(np.mean((g.to_numpy()[i] - p.to_numpy()[i]) ** 2)))
    return out


eski, yeni = tablo(sys.argv[1]), tablo(sys.argv[2])
ay = [m for m in eski if isinstance(m, (int, np.integer))]
print(" ".join(f"ay {m}: {eski[m]:.2f} → {yeni[m]:.2f} ({yeni[m] - eski[m]:+.2f})" for m in sorted(ay)))
print("ort fark", round(np.mean([yeni[m] - eski[m] for m in ay]), 3), "| iyi", sum(yeni[m] < eski[m] for m in ay), "/", len(ay))
print("havalimanı (6 ay RMSE farkı):", " ".join(f"{a} {np.sqrt(np.mean(yeni[a])) - np.sqrt(np.mean(eski[a])):+.2f}"
                                               for a in sorted(k for k in eski if isinstance(k, str))))

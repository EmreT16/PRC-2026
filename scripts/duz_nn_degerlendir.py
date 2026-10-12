"""10.10: Model A düzelticisinde LightGBM + TabM harmanı (CV, Oca / Tem; taban kqgpPeT2).
rhat = (1 − v) · rhat_LightGBM (ADSB_duz_cv_parkb_tabm2) + v · rhat_TabM (ADSB_duz_cv_parkb_tabm2_nn); taksi = max(60, NM − ofset + rhat).
Önceden tanımlı kabul: Oca / Tem ağırlıklı ≥ 0,10 iyi ve çapraz seçimde (bir ayda en iyi v → öbür ayda) iki ayda da iyi.
Kullanım: python scripts/duz_nn_degerlendir.py [nn=ADSB_duz_cv_parkb_tabm2_nn_e12]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PRC_HARMAN"] = "kqgpPeT2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config, validation as V  # noqa: E402

P = config.DATA_DIR / "preds"
VS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 11.10: 0,6 / 0,8 / 0,9 eklendi (tohum ortalamalı v sınaması)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = A.d_dw_split("split_1")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)  # nn=<dosya> (ör. ADSB_duz_cv_parkb_tabm2_nn_e12)
    for ad, f in (("rl", "ADSB_duz_cv_parkb_tabm2"), ("rn", opt.get("nn", "ADSB_duz_cv_parkb_tabm2_nn"))):
        d = d.merge(pd.read_parquet(P / f"{f}.parquet").rename(columns={"rhat": ad}), on="MVT_ID_mvt", how="left")
    d[["rl", "rn"]] = d[["rl", "rn"]].fillna(0.0)
    nm, yt, h = d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float), d.ofset_taban.to_numpy(float)
    ay = d.MVT_MONTH.to_numpy()
    iz = (d.rl.abs() > 1e-9).to_numpy()
    w = V.month_weights((1, 7))
    print(f"rhat korelasyonu (izli satırlar): {np.corrcoef(d.rl[iz], d.rn[iz])[0, 1]:.4f}")
    rows = {}
    for v in VS:
        p = np.maximum(60.0, nm - h + (1 - v) * d.rl.to_numpy() + v * d.rn.to_numpy())
        e = {a: np.mean((p[ay == a] - yt[ay == a]) ** 2) for a in (1, 7)}
        rows[v] = {"Oca": np.sqrt(e[1]), "Tem": np.sqrt(e[7]), "ağırlıklı": np.sqrt((w[1] * e[1] + w[7] * e[7]) / (w[1] + w[7])),
                   "izli": np.sqrt(np.mean((p[iz] - yt[iz]) ** 2))}
        for a in sorted(d.ADEP_mvt.unique()):
            m = (d.ADEP_mvt == a).to_numpy()
            rows[v][a] = np.sqrt(np.mean((p[m] - yt[m]) ** 2))
    t = pd.DataFrame(rows).T
    print(t.round(2).to_string())
    v1, v7 = t.loc[VS[1:], "Oca"].idxmin(), t.loc[VS[1:], "Tem"].idxmin()
    c7, c1 = t.loc[v1, "Tem"] - t.loc[0.0, "Tem"], t.loc[v7, "Oca"] - t.loc[0.0, "Oca"]
    en = t["ağırlıklı"].idxmin()
    kazanc = t.loc[en, "ağırlıklı"] - t.loc[0.0, "ağırlıklı"]
    kabul = kazanc <= -0.10 and c7 < 0 and c1 < 0
    print(f"çapraz: Oca'da v = {v1} → Tem {c7:+.3f} | Tem'de v = {v7} → Oca {c1:+.3f}; en iyi v = {en} ({kazanc:+.3f}) → "
          f"{'KABUL' if kabul else 'RET'}")
    if "ref_v" in opt:  # 11.10: v değişikliği kuralı — v* ≠ ref, ağırlıklı ≥ 0,05 iyi ve iki ayda da ref'ten iyi
        rv = float(opt["ref_v"])
        f = t.loc[en] - t.loc[rv]
        kv = en != rv and f["ağırlıklı"] <= -0.05 and f["Oca"] < 0 and f["Tem"] < 0
        print(f"v kuralı: v* = {en} ↔ ref {rv}: ağırlıklı {f['ağırlıklı']:+.3f}, Oca {f['Oca']:+.3f}, Tem {f['Tem']:+.3f} → "
              f"{'KABUL' if kv else 'RET'}")
        print(f"V_SECIM={en if kv else rv}")


if __name__ == "__main__":
    main()

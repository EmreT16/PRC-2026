"""10.10: iki TabM OOF'unun (aynı bölme) Model A harmanına katkısı, düzeltici sonrası (sabit rhat; tabm_degerlendir.py düzeni).
Taban: kqgpxg harman OOF'u (5 ağaç bileşeni), rhat ADSB_duz_cv_parkb (izsiz satırlarda kesin). ofset = (1 − w) · harman + w · nn.
Her TabM için ızgaradaki en iyi w'de Oca / Tem ağırlıklı RMSE; önceden tanımlı kural (TabM ayarı değişikliği): aday karşılıktan
≥ 0,05 iyi ve iki ayda da (kendi en iyi w'lerinde) iyi.
Kullanım: python scripts/tabm_pilot_kiyas.py tabm_kqgp_e12 tabm_kqgp_d1024 [split_1]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PRC_HARMAN"] = "kqgpxg"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config, validation as V  # noqa: E402

P = config.DATA_DIR / "preds"
W = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = sys.argv[1], sys.argv[2]
    sp = sys.argv[3] if len(sys.argv) > 3 else "split_1"
    d = A.d_dw_split(sp)[["MVT_ID_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]]
    d = d.merge(pd.read_parquet(P / "ADSB_duz_cv_parkb.parquet"), on="MVT_ID_mvt", how="left")
    for ad, f in (("na", a), ("nb", b)):
        d = d.merge(pd.read_parquet(P / f"NN_{f}_{sp}.parquet").rename(columns={"nn": ad}), on="MVT_ID_mvt", how="left",
                    validate="one_to_one")
    d["rhat"] = d["rhat"].fillna(0.0)
    nm, yt, h, r = (d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float),
                    d.ofset_taban.to_numpy(float), d.rhat.to_numpy(float))
    ay = d.MVT_MONTH.to_numpy()
    aylar = sorted(set(ay.tolist()))
    w8 = V.month_weights(tuple(aylar))
    print(f"{sp}: {len(d):,} satır | korelasyon {a} ↔ {b}: {np.corrcoef(d.na, d.nb)[0, 1]:.4f}")
    sonuc = {}
    for ad, col in ((a, "na"), (b, "nb")):
        n = d[col].to_numpy(float)
        tab = {}
        for w in W:
            p = np.maximum(60.0, nm - ((1 - w) * h + w * n) + r)
            e = {m: np.mean((p[ay == m] - yt[ay == m]) ** 2) for m in aylar}
            tab[w] = {**{f"ay{m}": np.sqrt(e[m]) for m in aylar},
                      "ağırlıklı": np.sqrt(sum(w8[m] * e[m] for m in aylar) / sum(w8[m] for m in aylar))}
        t = pd.DataFrame(tab).T
        en = t["ağırlıklı"].idxmin()
        sonuc[ad] = t.loc[en].rename(f"{ad} (w = {en})")
    s = pd.DataFrame(sonuc).T
    s.loc["fark"] = s.iloc[1] - s.iloc[0]
    print(s.round(3).to_string())
    f = s.loc["fark"]
    kabul = f["ağırlıklı"] <= -0.05 and all(f[f"ay{m}"] < 0 for m in aylar)
    print(f"ağırlıklı {f['ağırlıklı']:+.3f} → {'KABUL (tam koşu)' if kabul else 'RET'}")


if __name__ == "__main__":
    main()

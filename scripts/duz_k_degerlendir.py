"""11.10: rhat ölçeği k, düzelticiye TabM (v 0,7) + planli eklendikten
sonra yeniden (k = 1, LightGBM'li tek aile döneminde seçilmişti; iki ailenin ortalaması rhat'ı büzebilir).
taksi = max(60, NM − ofset + k · rhat); OOF ADSB_duz_cv_parkb_tabm2_lgbnn_planli (v353 / a56 düzeltici ayarları), Oca / Tem.
Önceden tanımlı kabul (v ızgarasıyla aynı): ağırlıklı ≥ 0,10 iyi ve çapraz seçimde (bir ayda en iyi k → öbür ayda) iki ayda da iyi.
Kullanım: python scripts/duz_k_degerlendir.py [oof=ADSB_duz_cv_parkb_tabm2_lgbnn_planli]
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
KS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    d = A.d_dw_split("split_1")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]]
    d = d.merge(pd.read_parquet(P / f"{opt.get('oof', 'ADSB_duz_cv_parkb_tabm2_lgbnn_planli')}.parquet")[["MVT_ID_mvt", "rhat"]],
                on="MVT_ID_mvt", how="left")
    d["rhat"] = d["rhat"].fillna(0.0)
    nm, yt, h, r = (d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float), d.ofset_taban.to_numpy(float),
                    d.rhat.to_numpy(float))
    ay = d.MVT_MONTH.to_numpy()
    w = V.month_weights((1, 7))
    rows = {}
    for k in KS:
        p = np.maximum(60.0, nm - h + k * r)
        e = {a: np.mean((p[ay == a] - yt[ay == a]) ** 2) for a in (1, 7)}
        rows[k] = {"Oca": np.sqrt(e[1]), "Tem": np.sqrt(e[7]), "ağırlıklı": np.sqrt((w[1] * e[1] + w[7] * e[7]) / (w[1] + w[7]))}
        for a in sorted(d.ADEP_mvt.unique()):
            m = (d.ADEP_mvt == a).to_numpy()
            rows[k][a] = np.sqrt(np.mean((p[m] - yt[m]) ** 2))
    t = pd.DataFrame(rows).T
    print(t.round(3).to_string())
    k1, k7 = t["Oca"].idxmin(), t["Tem"].idxmin()
    c7, c1 = t.loc[k1, "Tem"] - t.loc[1.0, "Tem"], t.loc[k7, "Oca"] - t.loc[1.0, "Oca"]
    en = t["ağırlıklı"].idxmin()
    kazanc = t.loc[en, "ağırlıklı"] - t.loc[1.0, "ağırlıklı"]
    kabul = en != 1.0 and kazanc <= -0.10 and c7 < 0 and c1 < 0
    print(f"çapraz: Oca'da k = {k1} → Tem {c7:+.3f} | Tem'de k = {k7} → Oca {c1:+.3f}; en iyi k = {en} ({kazanc:+.3f}) → "
          f"{'KABUL' if kabul else 'RET'}")


if __name__ == "__main__":
    main()

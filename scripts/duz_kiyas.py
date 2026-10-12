"""10.10: iki Model A düzeltici CV OOF'unun (Oca / Tem 2025, taban kqgpPeT2) karşılaştırması — aynı tablo düzeni
(scripts/duz_nn_degerlendir.py ile aynı hesap: taksi = max(60, NM − ofset_taban + rhat)).
Önceden tanımlı kabul (tek değişkenli düzeltici eki): Oca ve Tem ayrı ayrı iyi ve ağırlıklı ≥ 0,10 iyi.
Kullanım: python scripts/duz_kiyas.py ADSB_duz_cv_parkb_tabm2_lgbnn07 ADSB_duz_cv_parkb_tabm2_lgbnn_planli
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = sys.argv[1], sys.argv[2]
    d = A.d_dw_split("split_1")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]]
    for ad, f in (("ra", a), ("rb", b)):
        d = d.merge(pd.read_parquet(P / f"{f}.parquet").rename(columns={"rhat": ad}), on="MVT_ID_mvt", how="left")
    d[["ra", "rb"]] = d[["ra", "rb"]].fillna(0.0)
    nm, yt, h = d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float), d.ofset_taban.to_numpy(float)
    ay = d.MVT_MONTH.to_numpy()
    w = V.month_weights((1, 7))
    rows = {}
    for ad, r in (("karşılık", d.ra), ("aday", d.rb)):
        p = np.maximum(60.0, nm - h + r.to_numpy())
        e = {m: np.mean((p[ay == m] - yt[ay == m]) ** 2) for m in (1, 7)}
        rows[ad] = {"Oca": np.sqrt(e[1]), "Tem": np.sqrt(e[7]), "ağırlıklı": np.sqrt((w[1] * e[1] + w[7] * e[7]) / (w[1] + w[7]))}
        for apt in sorted(d.ADEP_mvt.unique()):
            m = (d.ADEP_mvt == apt).to_numpy()
            rows[ad][apt] = np.sqrt(np.mean((p[m] - yt[m]) ** 2))
    t = pd.DataFrame(rows).T
    t.loc["fark"] = t.loc["aday"] - t.loc["karşılık"]
    print(f"{a} ↔ {b}")
    print(t.round(3).to_string())
    f = t.loc["fark"]
    kabul = f["Oca"] < 0 and f["Tem"] < 0 and f["ağırlıklı"] <= -0.10
    print(f"Oca {f['Oca']:+.3f}, Tem {f['Tem']:+.3f}, ağırlıklı {f['ağırlıklı']:+.3f} → {'KABUL' if kabul else 'RET'}")


if __name__ == "__main__":
    main()

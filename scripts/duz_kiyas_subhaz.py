"""11.10: düzeltici CV OOF'larının Şubat / Haziran 2025'te
(sinama=2,6; her ay sırayla dışarıda, diğer 11 ayla eğitim) karşılaştırması. LB yalnız Oca / Tem 2026'yı ölçüyor; final
Şub / Haz 2026'yı da içeriyor → düzelticideki TabM ve planlı saatler bu aylarda da iyi mi?
taksi = max(60, NM − ofset_taban + rhat); Şub split_2'nin, Haz split_3'ün taban OOF'undan; aylar eşit ağırlıklı.
Önceden tanımlı yorum: aday iki ayda da karşılıktan iyi → doğrulandı; iki ayda da ≥ 0,10 kötü → finalde tartışılır;
arada → nötr (LB Oca / Tem kanıtı geçerli).
Kullanım: python scripts/duz_kiyas_subhaz.py <karşılık> <aday> [<aday2> …]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PRC_HARMAN"] = "kqgpPeT2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

P = config.DATA_DIR / "preds"
AY = {2: "Şub", 6: "Haz"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    adlar = sys.argv[1:]
    c = ["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]
    d = pd.concat([A.d_dw_split("split_2")[c], A.d_dw_split("split_3")[c]], ignore_index=True)
    d = d[d.MVT_MONTH.isin(list(AY))].reset_index(drop=True)
    for i, f in enumerate(adlar):
        d = d.merge(pd.read_parquet(P / f"{f}.parquet")[["MVT_ID_mvt", "rhat"]].rename(columns={"rhat": f"r{i}"}), on="MVT_ID_mvt",
                    how="left")
    nm, yt, h, ay = d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float), d.ofset_taban.to_numpy(float), d.MVT_MONTH.to_numpy()
    rows = {"taban (düzelticisiz)": None}
    for i, f in enumerate(adlar):
        rows[f] = d[f"r{i}"].fillna(0.0).to_numpy()
    sat = {}
    for ad, r in rows.items():
        p = np.maximum(60.0, nm - h + (0.0 if r is None else r))
        e = {m: np.mean((p[ay == m] - yt[ay == m]) ** 2) for m in AY}
        sat[ad] = {AY[m]: np.sqrt(e[m]) for m in AY} | {"ortalama": np.sqrt(np.mean(list(e.values())))}
        for apt in sorted(d.ADEP_mvt.unique()):
            m = (d.ADEP_mvt == apt).to_numpy()
            sat[ad][apt] = np.sqrt(np.mean((p[m] - yt[m]) ** 2))
    t = pd.DataFrame(sat).T
    print(t.round(3).to_string())
    ref = t.loc[adlar[0]]
    for f in adlar[1:]:
        fk = t.loc[f] - ref
        yorum = ("doğrulandı" if fk["Şub"] < 0 and fk["Haz"] < 0 else
                 "finalde tartışılır" if fk["Şub"] >= 0.10 and fk["Haz"] >= 0.10 else "nötr")
        print(f"{f} − {adlar[0]}: Şub {fk['Şub']:+.3f}, Haz {fk['Haz']:+.3f}, ortalama {fk['ortalama']:+.3f} → {yorum}")


if __name__ == "__main__":
    main()

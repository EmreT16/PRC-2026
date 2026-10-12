"""10.10: ADS-B izi olmayan satırlarda son tahmin doğrudan
harmandır (düzeltici yalnız izli satırlarda çalışır); izli satırlarda harman düzelticinin girdisidir. Harman ağırlıkları
(kqgpPeT2: 5 ağaç bileşeni × 0,6 + TabM 0,4) bütün satırlar için seçildi. İzsiz satırlar için ayrı ağırlık daha iyi mi?
Üretimde yalnız düzelticinin dokunmadığı satırlar (rhat = 0) değişir → düzeltici yeniden eğitilmez, izli satırlar aynen kalır.
Düzenek: 12 ay OOF (6 bölme; bileşenler g1_agirliklar_kqgpPeT2.json), izsiz = ADSB_ESLESME boş / 0 (adsb5 gün özellikleri).
İç içe ay dışarıda: her ay için ağırlıklar (≥ 0, toplam 1) kalan 11 ayın izsiz satırlarında SLSQP ile seçilir (amaç: 60 sn
tabanlı taksi SSE'si), dışarıdaki ayda ölçülür. Karşılık: üretim ağırlıkları.
Önceden tanımlı kabul: genel RMSE karşılığı (ΔMSE_izsiz · pay / (2 · 194)) toplamda ≥ 0,10 iyi ve ≥ 9/12 ay iyi.
Kullanım: python scripts/harman_izsiz_agirlik.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

os.environ.setdefault("PRC_ADSB_OZ", "adsb5")
os.environ["PRC_HARMAN"] = "kqgpPeT2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

P = config.DATA_DIR / "preds"
RMSE_GENEL = 194.0


def yukle():
    w = json.loads(A.AGIRLIK.read_text(encoding="utf-8"))
    parca = []
    for k in range(1, 7):
        d = pd.read_parquet(P / f"B1_split_{k}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y"])
        for ad in w["bilesenler"]:
            f, col = A.BILESEN[ad]
            b = pd.read_parquet(P / f"{f}_split_{k}.parquet", columns=["MVT_ID_mvt", col]).rename(columns={col: ad})
            d = d.merge(b, on="MVT_ID_mvt", how="left", validate="one_to_one")
        parca.append(d)
    d = pd.concat(parca, ignore_index=True)
    assert d[w["bilesenler"]].notna().all().all()
    fs = sorted(A.ADSB_DIR.glob("oz_2025*.parquet"))
    es = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "ADSB_ESLESME"]) for f in fs], ignore_index=True)
    es = es.drop_duplicates("MVT_ID_mvt")
    d = d.merge(es, on="MVT_ID_mvt", how="left")
    d["izsiz"] = ~(d["ADSB_ESLESME"].fillna(0) > 0)
    return d, w


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d, w = yukle()
    bil, w0 = w["bilesenler"], np.array(w["agirlik"], float)
    print(f"satır {len(d):,} | izsiz {int(d.izsiz.sum()):,} (%{100 * d.izsiz.mean():.1f}) | ADS-B gün dosyası ayları: "
          f"{sorted(d.loc[~d.izsiz, 'MVT_MONTH'].unique().tolist())}")
    print(d.groupby("ADEP_mvt").izsiz.mean().round(3).to_dict())
    u = d[d.izsiz].reset_index(drop=True)
    C = u[bil].to_numpy(float)
    nm, yt = u.NM_TAXI_DIFF_SEC.to_numpy(float), (u.NM_TAXI_DIFF_SEC - u.y).to_numpy(float)
    ay = u.MVT_MONTH.to_numpy()
    pay = {m: (ay == m).sum() / (d.MVT_MONTH == m).sum() for m in range(1, 13)}

    def sse(wv, i):
        e = np.maximum(60.0, nm[i] - C[i] @ wv) - yt[i]
        return float(e @ e)

    def sec(i):
        def f(wv):
            q = nm[i] - C[i] @ wv
            e = np.maximum(60.0, q) - yt[i]
            g = -2.0 * (C[i] * (e * (q > 60.0))[:, None]).sum(axis=0)
            return float(e @ e) / i.sum(), g / i.sum()
        r = minimize(f, w0, jac=True, method="SLSQP", bounds=[(0, 1)] * len(w0),
                     constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones_like(x)}],
                     options={"maxiter": 200, "ftol": 1e-10})
        return r.x
    tab = {}
    for m in range(1, 13):
        i_dis, i_ic = ay == m, ay != m
        wm = sec(i_ic)
        n = i_dis.sum()
        e0, e1 = sse(w0, i_dis) / n, sse(wm, i_dis) / n
        tab[m] = {"satır": n, "pay": pay[m], "üretim": np.sqrt(e0), "iç içe": np.sqrt(e1),
                  "genel karşılık": (e1 - e0) * pay[m] / (2 * RMSE_GENEL), **{b: wm[j] for j, b in enumerate(bil)}}
    t = pd.DataFrame(tab).T
    N = (d.MVT_MONTH.value_counts()).sort_index()
    top = float((t["genel karşılık"] * N.values).sum() / N.sum())
    iyi = int((t["iç içe"] < t["üretim"]).sum())
    print(t.round(3).to_string())
    wt = sec(np.ones(len(u), bool))
    print("12 ayla seçilen izsiz ağırlıklar:", {b: round(float(x), 4) for b, x in zip(bil, wt)}, "| üretim:", dict(zip(bil, w0)))
    print(f"izsiz RMSE (12 ay): üretim {np.sqrt(sse(w0, np.ones(len(u), bool)) / len(u)):.3f} → 12 ay seçimi "
          f"{np.sqrt(sse(wt, np.ones(len(u), bool)) / len(u)):.3f}")
    kabul = top <= -0.10 and iyi >= 9
    print(f"iç içe: genel RMSE karşılığı {top:+.3f}, iyi ay {iyi}/12 → {'KABUL' if kabul else 'RET'}")
    if kabul:
        json.dump({"bilesenler": bil, "agirlik": [round(float(x), 5) for x in wt],
                   "not": f"10.10: izsiz satırlar (düzelticinin dokunmadığı) için; iç içe {top:+.3f}, {iyi}/12 ay"},
                  open("kutuk/g1_agirliklar_kqgpPeT2_izsiz.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

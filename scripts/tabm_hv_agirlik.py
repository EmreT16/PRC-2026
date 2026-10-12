"""10.10: TabM'in harman ağırlığı havalimanına göre ayrılsa?
Gözlem: TabM'in katkısı izsiz havalimanlarında büyük (LTFM / LIRF / LFPG), izlilerde küçük ya da ters (EHAM).
Düzenek: 12 ay OOF (6 bölme; harman kqgpxg bileşenleri + NN_tabm_kqgp2), düzeltici rhat sabit (ADSB_duz_cv_parkb; izsiz
satırlarda kesin). İç içe ay dışarıda: dışarıdaki her ay için havalimanı başına w, kalan 11 aydan seçilir (ızgara), genel w ile
büzülür (λ: w_a = λ · w_genel + (1 − λ) · w_a*), dışarıdaki ayda ölçülür. Karşılık: tek w = 0,40 (üretim).
Kabul (önceden): toplam ≥ 0,10 iyi ve ≥ 9/12 ay iyi.
Kullanım: python scripts/tabm_hv_agirlik.py [ad=tabm_kqgp2]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PRC_HARMAN"] = "kqgpxg"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

P = config.DATA_DIR / "preds"
W = np.round(np.arange(0.0, 0.81, 0.1), 2)
W0 = 0.40


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    ad = opt.get("ad", "tabm_kqgp2")
    rh = pd.read_parquet(P / "ADSB_duz_cv_parkb.parquet")
    parca = []
    for k in range(1, 7):
        d = A.d_dw_split(f"split_{k}")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]]
        d = d.merge(pd.read_parquet(P / f"NN_{ad}_split_{k}.parquet"), on="MVT_ID_mvt", how="left", validate="one_to_one")
        d = d.merge(rh, on="MVT_ID_mvt", how="left", validate="one_to_one")
        parca.append(d)
    d = pd.concat(parca, ignore_index=True)
    d["rhat"] = d["rhat"].fillna(0.0)
    nm, yt = d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float)
    h, n, r = d.ofset_taban.to_numpy(float), d.nn.to_numpy(float), d.rhat.to_numpy(float)
    ay, apt = d.MVT_MONTH.to_numpy(), d.ADEP_mvt.to_numpy()
    # her (havalimanı, ay, w) için SSE tablosu
    sse = {}
    for w in W:
        e = (np.maximum(60.0, nm - ((1 - w) * h + w * n) + r) - yt) ** 2
        sse[w] = pd.Series(e).groupby([apt, ay]).sum()
    S = pd.DataFrame(sse)  # satır (apt, ay), sütun w
    S.index.names = ["apt", "ay"]
    N = pd.Series(np.ones(len(d))).groupby([apt, ay]).sum()
    aylar = sorted(set(ay.tolist()))
    print("havalimanı başına 12 ayda en iyi w (bilgi):", S.groupby("apt").sum().idxmin(axis=1).to_dict())
    for lam in (0.0, 0.5):
        tab = {}
        for m in aylar:
            egit = S[S.index.get_level_values("ay") != m].groupby("apt").sum()
            wa = egit.idxmin(axis=1)
            wa = (lam * W0 + (1 - lam) * wa).round(1).clip(W.min(), W.max())
            dis = S[S.index.get_level_values("ay") == m].droplevel("ay")
            n_m = N[N.index.get_level_values(1) == m].droplevel(1)
            s_ayri = sum(dis.loc[a, wa[a]] for a in dis.index)
            s_tek = dis[W0].sum()
            tab[m] = {"tek": np.sqrt(s_tek / n_m.sum()), "ayrı": np.sqrt(s_ayri / n_m.sum())}
        t = pd.DataFrame(tab).T
        t["fark"] = t["ayrı"] - t["tek"]
        top_tek = np.sqrt(sum(t.loc[m, "tek"] ** 2 * N[N.index.get_level_values(1) == m].sum() for m in aylar) / N.sum())
        top_ayri = np.sqrt(sum(t.loc[m, "ayrı"] ** 2 * N[N.index.get_level_values(1) == m].sum() for m in aylar) / N.sum())
        print(f"\nλ = {lam}: toplam tek w {top_tek:.3f} → havalimanı başına {top_ayri:.3f} ({top_ayri - top_tek:+.3f}); "
              f"iyi ay {int((t.fark < 0).sum())}/12")
        print(t.round(3).T.to_string())
    print("\n12 ayla seçilen havalimanı ağırlıkları (λ = 0 / 0,5):")
    wa = S.groupby("apt").sum().idxmin(axis=1)
    print(pd.DataFrame({"λ0": wa, "λ0,5": (0.5 * W0 + 0.5 * wa).round(2)}).T.to_string())


if __name__ == "__main__":
    main()

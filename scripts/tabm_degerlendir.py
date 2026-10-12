"""09.10 akşam: TabM OOF'unun Model A harmanına katkısı (ay dışı bölme).

Taban: üretim harmanının OOF'u (HARMAN_kqgpxg_<bölme>: kqgp bileşenleri, g1_agirliklar_kqgpxg ağırlıkları; testte aynı
ağırlıklar final bileşenleriyle). TabM: NN_<ad>_<bölme> (nn = ofset tahmini). Karışım: ofset = (1 − w) · harman + w · nn.
w çapraz seçilir (bölmenin bir ayında en iyi w → öbür ayda ölçülür); ayrıca sabit w ızgarası.
Düzeltici sonrası (yaklaşık): ADSB_duz_cv_parkb'nin rhat'ı aynen eklenir (taksi = max(60, NM − ofset + rhat)); izsiz
satırlarda rhat = 0 olduğundan orada kesin, izli satırlarda düzeltici yeniden eğitilmediği için yaklaşık.
Kullanım: python scripts/tabm_degerlendir.py tabm_kqgp split_1 [split_2 ...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PRC_HARMAN", "kqgpxg")  # üretim düzelticisinin OOF tabanı (10.10: ortamdan değiştirilebilir, ör. kqgpPeT2)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

P = config.DATA_DIR / "preds"
W = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ad, bolmeler = sys.argv[1], sys.argv[2:]
    rh = pd.read_parquet(P / f"{os.environ.get('TD_RHAT', 'ADSB_duz_cv_parkb')}.parquet")  # 10.10: TD_RHAT ile başka düzeltici OOF'u
    hepsi = []
    for sp in bolmeler:
        d = A.d_dw_split(sp)[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]]
        d = d.rename(columns={"ofset_taban": "harman"})  # HARMAN_kqgpxg_<bölme> ile aynı (split_1'de denetlendi); 4–6'da dosya yok
        d = d.merge(pd.read_parquet(P / f"NN_{ad}_{sp}.parquet"), on="MVT_ID_mvt", how="left", validate="one_to_one")
        d = d.merge(rh, on="MVT_ID_mvt", how="left", validate="one_to_one")
        assert d[["harman", "nn"]].notna().all().all()
        d["rhat"] = d["rhat"].fillna(0.0)
        nm, yt = d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float)
        h, n, r = d.harman.to_numpy(float), d.nn.to_numpy(float), d.rhat.to_numpy(float)
        ay = d.MVT_MONTH.to_numpy()
        iz = np.abs(r) > 1e-9
        print(f"\n=== {sp}: {len(d):,} satır | korelasyon (ofset) harman ↔ {ad} {np.corrcoef(h, n)[0, 1]:.4f}")
        for etiket, rr in (("düzeltici öncesi", np.zeros_like(r)), ("düzeltici sonrası", r)):
            tab = []
            for w in W + [1.0]:
                p = np.maximum(60.0, nm - ((1 - w) * h + w * n) + rr)
                row = {"w": w, "toplam": rmse(p, yt)}
                for m in sorted(set(ay.tolist())):
                    row[f"ay{m}"] = rmse(p[ay == m], yt[ay == m])
                row["izsiz"] = rmse(p[~iz], yt[~iz])
                row["izli"] = rmse(p[iz], yt[iz])
                tab.append(row)
            t = pd.DataFrame(tab).set_index("w")
            print(f"--- {etiket} (izsiz {int((~iz).sum()):,} / izli {int(iz.sum()):,} satır)")
            print(t.round(3).to_string())
            aylar = [c for c in t.columns if c.startswith("ay")]
            if len(aylar) == 2:  # çapraz: bir ayda seç, öbüründe ölç
                cap = []
                for a, b in (aylar, aylar[::-1]):
                    w = t.loc[W, a].idxmin()
                    cap.append(f"{a}'da seçilen w = {w} → {b}: {t.loc[0.0, b]:.3f} → {t.loc[w, b]:.3f} ({t.loc[w, b] - t.loc[0.0, b]:+.3f})")
                print("çapraz:", " | ".join(cap))
        # havalimanı başına (düzeltici sonrası, w = 0,2)
        p0 = np.maximum(60.0, nm - h + r)
        p1 = np.maximum(60.0, nm - (0.8 * h + 0.2 * n) + r)
        g = pd.DataFrame({"apt": d.ADEP_mvt, "e0": (p0 - yt) ** 2, "e1": (p1 - yt) ** 2})
        s = g.groupby("apt").agg(n=("e0", "size"), taban=("e0", "mean"), w02=("e1", "mean"))
        s = np.sqrt(s[["taban", "w02"]]).assign(n=s.n, fark=lambda x: x.w02 - x.taban)
        print("--- havalimanı başına, düzeltici sonrası, w = 0,2")
        print(s.round(2).to_string())
        hepsi.append(pd.DataFrame({"ay": ay, "nm": nm, "yt": yt, "h": h, "n": n, "r": r}))
    if len(hepsi) > 1:  # bütün bölmeler birlikte: 12 ay, düzeltici sonrası
        t = pd.concat(hepsi, ignore_index=True)
        print(f"\n=== {len(bolmeler)} bölme birlikte ({len(t):,} satır), düzeltici sonrası; ay başına RMSE farkı (w − taban)")
        rows = {}
        for w in W[1:]:
            p = np.maximum(60.0, t.nm - ((1 - w) * t.h + w * t.n) + t.r)
            p0 = np.maximum(60.0, t.nm - t.h + t.r)
            e, e0 = (p - t.yt) ** 2, (p0 - t.yt) ** 2
            fa = np.sqrt(e.groupby(t.ay).mean()) - np.sqrt(e0.groupby(t.ay).mean())
            rows[w] = {**{f"ay{m}": v for m, v in fa.items()}, "toplam": np.sqrt(e.mean()) - np.sqrt(e0.mean()),
                       "iyi_ay": int((fa < 0).sum())}
        print(f"taban toplam RMSE {np.sqrt(((np.maximum(60.0, t.nm - t.h + t.r) - t.yt) ** 2).mean()):.3f}")
        print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()

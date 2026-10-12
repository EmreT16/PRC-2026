"""iter_egrisi.py çıktısının özeti (04.10 gece): bölme başına Oca/Tem ağırlıklı RMSE eğrisi, genel en iyi tur, şimdiki turla
fark ve havalimanı bazında en iyi tur (bölmeler toplamında seçilir; her bölmedeki en iyi tur kararlılık için yanında).
Kullanım: python scripts/iter_ozet.py <lgb|xgb>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

SIMDI = {"lgbh3": (0, 750), "e1": (400, 750), "xgb": (0, 1500)}


def katki(x):
    """(split, apt, kc, k) başına ay ağırlıklı MSE katkısı; havalimanları üzerinden toplanınca bölmenin ağırlıklı MSE'si."""
    parca = []
    for s, g in x.groupby("split"):
        _, _, vm = V.split_masks(pd.Series(np.arange(1, 13)), s)
        w = V.month_weights(vm)
        n_ay = g[(g.kc == g.kc.iloc[0]) & (g.k == g.k.iloc[0])].groupby("ay").n.sum()
        g = g.assign(c=g.sse / g.ay.map(n_ay) * g.ay.map(w))
        parca.append(g.groupby(["split", "apt", "kc", "k"], as_index=False).c.sum())
    return pd.concat(parca, ignore_index=True)


def ozet(bil, x):
    c = katki(x)
    s = c.groupby(["split", "kc", "k"]).c.sum().pipe(np.sqrt).unstack("split")
    s["ort"] = s.mean(axis=1)
    kc0, k0 = SIMDI[bil]
    en = s["ort"].idxmin()
    print(f"\n=== {bil}: şimdiki (kc={kc0}, k={k0}) {s.loc[(kc0, k0), 'ort']:.3f} | en iyi (kc={en[0]}, k={en[1]}) "
          f"{s.loc[en, 'ort']:.3f} | fark {s.loc[en, 'ort'] - s.loc[(kc0, k0), 'ort']:+.3f}")
    print("bölme başına en iyi:", {sp: s[sp].idxmin() for sp in s.columns if sp != "ort"})
    kc_en = en[0]
    egri = s.xs(kc_en, level="kc")["ort"]
    print("eğri (kc=%d): " % kc_en + " ".join(f"{k}:{v:.2f}" for k, v in egri.items() if k % 250 == 0 or k == 100))
    if bil == "e1":
        print("kc eğrisi (k=en iyi): " + " ".join(f"{kc}:{v:.2f}" for kc, v in s.xs(en[1], level="k")["ort"].items()))
    # havalimanı bazında
    a = c.groupby(["apt", "kc", "k"]).c.sum()
    sat = []
    for apt, g in a.groupby("apt"):
        g = g.droplevel("apt")
        b = g.idxmin()
        bs = {sp: c[(c.apt == apt) & (c.split == sp)].set_index(["kc", "k"]).c.idxmin()[1] for sp in sorted(c.split.unique())}
        sat.append((apt, b[0], b[1], g.loc[en] - g.loc[b], g.loc[(kc0, k0)] - g.loc[en], bs))
    t = pd.DataFrame(sat, columns=["apt", "kc", "k", "mse_kazanc_hv", "mse_kazanc_genel", "bolme_k"])
    n_sp = c.split.nunique()
    mse_en = (s.loc[en, [sp for sp in s.columns if sp != "ort"]] ** 2).mean()
    print(t.to_string(index=False))
    print(f"havalimanı bazında seçimle ek kazanç (RMSE, bölme ortalaması yaklaşık): "
          f"{np.sqrt(mse_en - t.mse_kazanc_hv.sum() / n_sp) - np.sqrt(mse_en):+.3f}")


def capraz(bil, x):
    """İç içe kontrol: turu 2 bölmede seç (genel ya da havalimanı bazında), dışarıdaki bölmede ölç."""
    c = katki(x)
    kc0, k0 = SIMDI[bil]
    sat = []
    for s in sorted(c.split.unique()):
        ic, dis = c[c.split != s], c[c.split == s].set_index(["apt", "kc", "k"]).c
        g_en = ic.groupby(["kc", "k"]).c.sum().idxmin()
        a_en = ic.groupby(["apt", "kc", "k"]).c.sum().groupby("apt").idxmin()
        simdi = np.sqrt(dis.xs((kc0, k0), level=["kc", "k"]).sum())
        genel = np.sqrt(dis.xs(g_en, level=["kc", "k"]).sum())
        hv = np.sqrt(sum(dis.loc[t] for t in a_en))
        sat.append((s, f"{g_en}", round(genel - simdi, 3), round(hv - simdi, 3)))
    t = pd.DataFrame(sat, columns=["dışarıdaki", "genel seçim", "genel − şimdiki", "havalimanı − şimdiki"])
    print(f"--- {bil} iç içe kontrol")
    print(t.to_string(index=False))
    print(f"ortalama: genel {t['genel − şimdiki'].mean():+.3f} | havalimanı bazında {t['havalimanı − şimdiki'].mean():+.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = pd.read_parquet(config.ROOT / "kutuk" / f"iter_egrisi_{sys.argv[1]}.parquet")
    for bil, x in d.groupby("bilesen"):
        ozet(bil, x)
        capraz(bil, x)

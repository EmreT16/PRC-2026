"""Harman ağırlıklarının 6 bölmede yeniden seçimi (05.10). Eğitim yok, yalnız OOF.
Ağırlıklar 5 bölmede seçilir (negatif olmayan, toplamı 1; ay ağırlıklı MSE, SLSQP), dışarıdaki bölmede şimdiki
ağırlıklara (D_dw: xgb 0,1021 / gcb 0,2036 / e1h3 0,2211 / lgbh3 0,1236 / dw 0,3497) karşı ölçülür. Hata 60 sn alt sınırlı taksi.
Kullanım: python scripts/harman_agirlik6.py <kqgpy|kqgp>
Son satır: 6 bölmenin tamamında seçilen ağırlık (önerilen)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

P = config.DATA_DIR / "preds"
AD = ["xgb", "gcb", "e1h3", "lgbh3", "dw"]
W0 = np.array([0.1021, 0.2036, 0.2211, 0.1236, 0.3497])


def dosyalar(e):
    return [(f"ADAY_v_xgb_{e}_1500", "harman"), (f"GCB_it12000_ort2_{e}", "global_cb"),
            (f"ADAY_v_e1_huber1000_3t_{e}", "harman"), (f"ADAY_v_lgb_huber1000_3t_{e}", "harman"),
            (f"GCB_dw_lr03_ek5_l2_10_{e}", "global_cb")]


def yukle(e, s):
    d = pd.read_parquet(P / f"B1_{s}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y"])
    X = []
    for f, c in dosyalar(e):
        b = d[["MVT_ID_mvt"]].merge(pd.read_parquet(P / f"{f}_{s}.parquet", columns=["MVT_ID_mvt", c]), on="MVT_ID_mvt",
                                    how="left", validate="one_to_one")[c].to_numpy(float)
        assert np.isfinite(b).all(), (f, s)
        X.append(b)
    _, _, vm = V.split_masks(pd.Series(np.arange(1, 13)), s)
    w = V.month_weights(vm)
    m = d["MVT_MONTH"].to_numpy()
    n_ay = pd.Series(m).value_counts()
    sw = pd.Series(m).map(lambda a: w[a] / n_ay[a]).to_numpy(float)  # satır ağırlığı: ay ağırlığı / ay satır sayısı
    return np.column_stack(X), d["y"].to_numpy(float), d["NM_TAXI_DIFF_SEC"].to_numpy(float), sw, d["ADEP_mvt"].to_numpy()


def mse(w, X, y, nm, sw, floor=True):
    p = X @ w
    if floor:
        p = V.floor_offset(p, nm)
    return float(np.sum(sw * (p - y) ** 2))


def sec(parca):
    def f(w):
        return sum(mse(w, X, y, nm, sw, floor=False) for X, y, nm, sw, _ in parca)
    r = minimize(f, W0, method="SLSQP", bounds=[(0, 1)] * len(W0),
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 200, "ftol": 1e-12})
    return r.x


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    e = sys.argv[1]
    S = [f"split_{k}" for k in range(1, 7)]
    veri = {s: yukle(e, s) for s in S}
    sat = []
    for s in S:
        w = sec([veri[t] for t in S if t != s])
        X, y, nm, sw, _ = veri[s]
        r0, r1 = np.sqrt(mse(W0, X, y, nm, sw)), np.sqrt(mse(w, X, y, nm, sw))
        sat.append((s, round(r0, 3), round(r1, 3), round(r1 - r0, 3), " ".join(f"{a}={v:.3f}" for a, v in zip(AD, w))))
    t = pd.DataFrame(sat, columns=["dışarıdaki", "şimdiki", "yeni", "fark", "ağırlıklar (5 bölmede seçilen)"])
    print(f"=== {e}: ağırlıklar 5 bölmede seçilip 6.'da ölçüldü\n" + t.to_string(index=False))
    print(f"ortalama fark {t['fark'].mean():+.3f} | iyileşen bölme {(t['fark'] < 0).sum()}/6")
    w = sec(list(veri.values()))
    print("6 bölmenin tamamıyla seçilen:", " ".join(f"{a}={v:.4f}" for a, v in zip(AD, w)))
    # havalimanı bazında fark (6 bölme toplamı, yeni − şimdiki MSE katkısı)
    h = {}
    for s in S:
        X, y, nm, sw, apt = veri[s]
        e0 = sw * (V.floor_offset(X @ W0, nm) - y) ** 2
        e1 = sw * (V.floor_offset(X @ w, nm) - y) ** 2
        for a in np.unique(apt):
            h[a] = h.get(a, 0.0) + float((e1 - e0)[apt == a].sum()) / len(S)
    print("havalimanı MSE katkı farkı (yeni − şimdiki):", {a: round(v, 1) for a, v in h.items()})


def hv_capraz(veri, S, kuculme=0.0):
    """Havalimanı bazında ağırlık (05.10): her havalimanında 5 bölmede seç, 6.'da ölç; genel (5 bölmede seçilen) ağırlığa
    karşı. kuculme: havalimanı ağırlığını genel ağırlığa doğru çekme payı (0 = saf havalimanı)."""
    sat = []
    for s in S:
        ic = [veri[t] for t in S if t != s]
        wg = sec(ic)
        X, y, nm, sw, apt = veri[s]
        p_g = V.floor_offset(X @ wg, nm)
        p_h = p_g.copy()
        for a in np.unique(apt):
            parca = [(Xi[ai == a], yi[ai == a], nmi[ai == a], swi[ai == a], ai[ai == a]) for Xi, yi, nmi, swi, ai in ic]
            wa = (1 - kuculme) * sec(parca) + kuculme * wg
            m = apt == a
            p_h[m] = V.floor_offset(X[m] @ wa, nm[m])
        r_g, r_h = np.sqrt(np.sum(sw * (p_g - y) ** 2)), np.sqrt(np.sum(sw * (p_h - y) ** 2))
        sat.append((s, round(r_g, 3), round(r_h, 3), round(r_h - r_g, 3)))
    t = pd.DataFrame(sat, columns=["dışarıdaki", "genel", "havalimanı", "fark"])
    print(f"--- havalimanı bazında (küçülme {kuculme}) genel ağırlığa göre\n" + t.to_string(index=False))
    print(f"ortalama {t['fark'].mean():+.3f} | iyileşen {(t['fark'] < 0).sum()}/6")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "hv":
    for k in (0.0, 0.5):
        hv_capraz(veri, S, k)

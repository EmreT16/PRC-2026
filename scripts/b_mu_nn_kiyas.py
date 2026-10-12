"""11.10: B μ TabM OOF'unun ayar varyantlarını mevcut OOF'la karşılaştırır.
Düzen scripts/b_mu_nn_degerlendir.py ile aynı (μ = (1 − w) · μ_mevcut + w · μ_TabM, combine_tum, tavan 5.000; 24 saat satırları
hariç; 8 havalimanı). Referans: mevcut TabM OOF (_dparkktahkomgspcyte_e12_t1) w = 0,6'da (üretim ağırlığı).
Kontrol: aynı ayarlarla başka tohum (gürültü düzeyi; kural eşiğini belirler). Önceden tanımlı kabul (her aday için):
  aday kendi en iyi w'sinde 8 hav. RMSE ≤ ref − eşik, ≥ 3/4 çeyrek ve ≥ 5/8 havalimanı ref'ten iyi;
  eşik = max(0,5, 2 · |kontrol − ref|) (kontrol de kendi en iyi w'sinde).
Kullanım: python scripts/b_mu_nn_kiyas.py kontrol=<ek> <aday_ek> [<aday_ek> …]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

APT8 = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM", "EHAM", "LSZH"]
K = ["mu", "p24", "yt_tum", "d", "d24"]
W = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REF = "_dparkktahkomgspcyte_e12_t1"


def main():
    kontrol = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("kontrol=")), None)
    adaylar = [a for a in sys.argv[1:] if "=" not in a]
    b = MB.load_b("train")
    df = b[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B]].copy()
    df["r"] = MB.regimes(b)
    oku = lambda f: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f), on="MVT_ID_mvt", how="left")  # noqa: E731
    P, C = oku("BM_cv_tum_dparkktah_oof.parquet"), oku("BM_cv_tum_dparkktahkomgspcyte_cb_oof.parquet")
    y = df[MB.TARGET_B].to_numpy(float)
    mu0 = 0.5 * C["bilesen_mu"].to_numpy(float) + 0.5 * C["bilesen_mu_cb"].to_numpy(float)
    m8 = (df.ADEP_mvt.isin(APT8) & (df.r != 2)).to_numpy()
    n0 = m8 & (df.r == 0).to_numpy()

    def olc(ek, w):
        mun = oku(f"BM_cv_mu_nn{ek}_oof.parquet")["bilesen_mu_nn"].to_numpy(float)
        assert np.isfinite(mun[m8]).all()
        c = {k: P[f"bilesen_{k}"].to_numpy(float) for k in K}
        c["mu"] = (1 - w) * mu0 + w * mun
        e2 = (MB.combine_tum(c, df["ADEP_mvt"].to_numpy(), yt="yt_tum", cap_non_lirf=5000.0) - y) ** 2
        r = {"8 hav.": np.sqrt(e2[m8].mean()), "μ RMSE": np.sqrt(((mun - y) ** 2)[n0].mean())}
        for q, ms in V.QUARTERS.items():
            r[q] = np.sqrt(e2[m8 & df.MVT_MONTH.isin(ms).to_numpy()].mean())
        for a in APT8:
            r[a] = np.sqrt(e2[m8 & (df.ADEP_mvt == a).to_numpy()].mean())
        return pd.Series(r)

    def en_iyi(ek):
        t = pd.DataFrame({w: olc(ek, w) for w in W}).T
        w = t["8 hav."].idxmin()
        return w, t.loc[w]

    ref = olc(REF, 0.6)
    print(f"ref {REF} (w 0,6): 8 hav. {ref['8 hav.']:.2f}, μ RMSE {ref['μ RMSE']:.1f}")
    esik = 0.5
    if kontrol:
        wk, k = en_iyi(kontrol)
        esik = max(0.5, 2 * abs(k["8 hav."] - ref["8 hav."]))
        print(f"kontrol {kontrol} (en iyi w {wk}): 8 hav. {k['8 hav.']:.2f} (fark {k['8 hav.'] - ref['8 hav.']:+.2f}), "
              f"μ RMSE {k['μ RMSE']:.1f} → eşik {esik:.2f}")
    sat = {"ref": ref}
    for ek in adaylar:
        w, a = en_iyi(ek)
        sat[f"{ek} (w {w})"] = a
        q = int((a[list(V.QUARTERS)] < ref[list(V.QUARTERS)]).sum())
        h = int((a[APT8] < ref[APT8]).sum())
        d = a["8 hav."] - ref["8 hav."]
        kabul = d <= -esik and q >= 3 and h >= 5
        print(f"{ek}: en iyi w {w}, 8 hav. {a['8 hav.']:.2f} ({d:+.2f}), çeyrek {q}/4, havalimanı {h}/8 → {'KABUL' if kabul else 'RET'}")
    print(pd.DataFrame(sat).T.round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

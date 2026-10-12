"""10.10: Model B μ'ye TabM katkısı (CV, OOF bileşenleriyle). Üretim düzeni (v316 / v334 melezi): p24 / yt / d park + ktah
modelinden (BM_cv_tum_dparkktah_oof), μ = 0,5 · LightGBM + 0,5 · CatBoost (BM_cv_tum_dparkktahkomgspcyte_cb_oof; üretimde
CatBoost iki çalıştırmanın ortalaması), tavan 5.000. Aday: μ = (1 − w) · μ_mevcut + w · μ_TabM (BM_cv_mu_nn<ek>_oof).
Ölçü: 24 saat satırları hariç; 6 üretim havalimanı (EDDM, EGLL, LEBL, LEMD, LFPG, LTFM) ve 8 (+ EHAM, LSZH); çeyrekler ve
havalimanları. Önceden tanımlı kabul: 8 havalimanında en iyi w, ≥ 3/4 çeyrek ve ≥ 5/8 havalimanı iyi.
Kullanım: python scripts/b_mu_nn_degerlendir.py _dparkktahkomgspcyte_e12_t1
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

APT6 = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
APT8 = APT6 + ["EHAM", "LSZH"]
K = ["mu", "p24", "yt_tum", "d", "d24"]
W = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10.10 gece: LB v347 −0,858 sonrası ızgara genişletildi


def main(ek):
    b = MB.load_b("train")
    df = b[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B, MB.DELAY]].copy()
    df["r"] = MB.regimes(b)
    oku = lambda f: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f), on="MVT_ID_mvt", how="left")  # noqa: E731
    P, C, N = (oku("BM_cv_tum_dparkktah_oof.parquet"), oku("BM_cv_tum_dparkktahkomgspcyte_cb_oof.parquet"),
               oku(f"BM_cv_mu_nn{ek}_oof.parquet"))
    y = df[MB.TARGET_B].to_numpy(float)
    mu0 = 0.5 * C["bilesen_mu"].to_numpy(float) + 0.5 * C["bilesen_mu_cb"].to_numpy(float)
    mun = N["bilesen_mu_nn"].to_numpy(float)
    m8 = (df.ADEP_mvt.isin(APT8) & (df.r != 2)).to_numpy()
    m6 = (df.ADEP_mvt.isin(APT6) & (df.r != 2)).to_numpy()
    assert np.isfinite(mun[m8]).all() and np.isfinite(mu0[m8]).all()
    n0 = m8 & (df.r == 0).to_numpy()
    print(f"μ (normal rejim, 8 havalimanı): mevcut RMSE {np.sqrt(((mu0 - y) ** 2)[n0].mean()):.1f}, TabM "
          f"{np.sqrt(((mun - y) ** 2)[n0].mean()):.1f}, korelasyon {np.corrcoef(mu0[n0], mun[n0])[0, 1]:.4f}")
    sat = {}
    for w in W:
        c = {k: P[f"bilesen_{k}"].to_numpy(float) for k in K}
        c["mu"] = (1 - w) * mu0 + w * mun
        e2 = (MB.combine_tum(c, df["ADEP_mvt"].to_numpy(), yt="yt_tum", cap_non_lirf=5000.0) - y) ** 2
        r = {"8 hav.": np.sqrt(e2[m8].mean()), "6 hav.": np.sqrt(e2[m6].mean())}
        for q, ms in V.QUARTERS.items():
            r[q] = np.sqrt(e2[m8 & df.MVT_MONTH.isin(ms).to_numpy()].mean())
        for a in APT8:
            r[a] = np.sqrt(e2[m8 & (df.ADEP_mvt == a).to_numpy()].mean())
        for ay in (2, 6):
            r[f"ay{ay}"] = np.sqrt(e2[m8 & (df.MVT_MONTH == ay).to_numpy()].mean())
        sat[w] = r
    t = pd.DataFrame(sat).T
    print(t.round(1).to_string())
    en = t["8 hav."].idxmin()
    q_iyi = int((t.loc[en, list(V.QUARTERS)] < t.loc[0.0, list(V.QUARTERS)]).sum())
    a_iyi = int((t.loc[en, APT8] < t.loc[0.0, APT8]).sum())
    kabul = en > 0 and q_iyi >= 3 and a_iyi >= 5
    print(f"en iyi w = {en}: 8 hav. {t.loc[0.0, '8 hav.']:.1f} → {t.loc[en, '8 hav.']:.1f}, çeyrek {q_iyi}/4, havalimanı {a_iyi}/8 "
          f"→ {'KABUL' if kabul else 'RET'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])

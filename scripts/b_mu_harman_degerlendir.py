"""08.10: Model B μ harmanı (LightGBM μ + CatBoost μ) CV değerlendirmesi, OOF bileşenleriyle. v295'teki melez düzen:
p24 / yt / d park + ktah modelinden (BM_cv_tum_dparkktah_oof), μ = a · μ_lgb + (1 − a) · μ_cb (kaynak: <ek>_oof).
Ölçü b_park_degerlendir.py ile aynı (6 havalimanı, 24 sa hariç, tum_mu_yt_tavan3500 birleşimi).
Kullanım: python scripts/b_mu_harman_degerlendir.py _dparkktahkomgspcyte_cb"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

APT6 = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
K = ["mu", "p24", "yt_tum", "d", "d24"]


def main(ek):
    df = MB.load_b("train")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B, MB.DELAY]]
    df["r"] = MB.regimes(MB.load_b("train"))
    oku = lambda e: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f"BM_cv_tum{e}_oof.parquet"),  # noqa: E731
                                             on="MVT_ID_mvt", how="left")
    P, C = oku("_dparkktah"), oku(ek)
    m = (df.ADEP_mvt.isin(APT6) & (df.r != 2)).to_numpy()
    y = df[MB.TARGET_B].to_numpy(float)
    mu_l, mu_c = C["bilesen_mu"].to_numpy(float), C["bilesen_mu_cb"].to_numpy(float)
    print(f"μ korelasyonu (6 havalimanı, normal rejim): {np.corrcoef(mu_l[m & (df.r == 0).to_numpy()], mu_c[m & (df.r == 0).to_numpy()])[0, 1]:.4f}")
    sat = {}
    for ad, mu in (("park", P["bilesen_mu"].to_numpy(float)), ("μ lgb (v295)", mu_l), ("μ cb", mu_c),
                   ("0,7 lgb + 0,3 cb", 0.7 * mu_l + 0.3 * mu_c), ("0,5 / 0,5", 0.5 * mu_l + 0.5 * mu_c),
                   ("0,3 lgb + 0,7 cb", 0.3 * mu_l + 0.7 * mu_c)):
        c = {k: P[f"bilesen_{k}"].to_numpy(float) for k in K}
        c["mu"] = mu
        e2 = (MB.combine_tum(c, df["ADEP_mvt"].to_numpy(), yt="yt_tum", cap_non_lirf=3500.0) - y) ** 2
        r = {"6 hav.": np.sqrt(e2[m].mean()), "μ normal": np.sqrt(((mu - y) ** 2)[m & (df.r == 0).to_numpy()].mean())}
        for q, ms in V.QUARTERS.items():
            r[q] = np.sqrt(e2[m & df.MVT_MONTH.isin(ms).to_numpy()].mean())
        for a in APT6:
            r[a] = np.sqrt(e2[m & (df.ADEP_mvt == a).to_numpy()].mean())
        sat[ad] = r
    print(pd.DataFrame(sat).round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])

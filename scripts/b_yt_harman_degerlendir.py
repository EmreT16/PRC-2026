"""08.10: Model B yer tutucu olasılığı (yt) için LightGBM + CatBoost CV değerlendirmesi, OOF bileşenleriyle. v297 düzeni:
μ = 0,5 · LightGBM + 0,5 · CatBoost (BM_cv_tum_dparkktahkomgspcyte_cb_oof), p24 / d park + ktah modelinden;
yt = a · yt_lgb (park + ktah) + (1 − a) · yt_cb (<ek>_oof). Ölçü b_park_degerlendir.py ile aynı (6 havalimanı, 24 sa hariç).
Kullanım: python scripts/b_yt_harman_degerlendir.py _dparkktah_ytcb"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

APT6 = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]


def main(ek):
    df = MB.load_b("train")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B, MB.DELAY]]
    df["r"] = MB.regimes(MB.load_b("train"))
    oku = lambda e: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f"BM_cv_tum{e}_oof.parquet"),  # noqa: E731
                                             on="MVT_ID_mvt", how="left")
    P, M, C = oku("_dparkktah"), oku("_dparkktahkomgspcyte_cb"), oku(ek)
    m = (df.ADEP_mvt.isin(APT6) & (df.r != 2)).to_numpy()
    y = df[MB.TARGET_B].to_numpy(float)
    yt_l, yt_c = P["bilesen_yt_tum"].to_numpy(float), C["bilesen_yt_tum_cb"].to_numpy(float)
    hedef = (df.r == 1).to_numpy()
    k = m & (df.r != 2).to_numpy()
    for ad, p in (("lgb", yt_l), ("cb", yt_c), ("0,5 / 0,5", 0.5 * yt_l + 0.5 * yt_c)):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        print(f"yt {ad}: logloss {-np.mean(hedef[k] * np.log(p[k]) + (1 - hedef[k]) * np.log(1 - p[k])):.4f}")
    sat = {}
    for ad, yt in (("v297 (yt lgb)", yt_l), ("yt cb", yt_c), ("yt 0,5 / 0,5", 0.5 * yt_l + 0.5 * yt_c)):
        c = {k_: P[f"bilesen_{k_}"].to_numpy(float) for k_ in ("p24", "d", "d24")}
        c["mu"] = 0.5 * M["bilesen_mu"].to_numpy(float) + 0.5 * M["bilesen_mu_cb"].to_numpy(float)
        c["yt_tum"] = yt
        e2 = (MB.combine_tum(c, df["ADEP_mvt"].to_numpy(), yt="yt_tum", cap_non_lirf=3500.0) - y) ** 2
        r = {"6 hav.": np.sqrt(e2[m].mean())}
        for q, ms in V.QUARTERS.items():
            r[q] = np.sqrt(e2[m & df.MVT_MONTH.isin(ms).to_numpy()].mean())
        for a in APT6:
            r[a] = np.sqrt(e2[m & (df.ADEP_mvt == a).to_numpy()].mean())
        sat[ad] = r
    print(pd.DataFrame(sat).round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])

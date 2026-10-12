"""08.10: Model B'de bir bileşen (yt_tum ya da p24) için LightGBM ↔ CatBoost ↔ yarı yarıya, OOF bileşenleriyle. v297 düzeni:
μ = 0,5 · LightGBM + 0,5 · CatBoost (BM_cv_tum_dparkktahkomgspcyte_cb_oof), diğerleri park + ktah modelinden; yalnız
seçilen bileşen değişir (CatBoost'u <ek>_oof'taki bilesen_<bileşen>_cb). Ölçü: 6 havalimanı, 24 sa hariç ve dahil.
Kullanım: python scripts/b_cb_bilesen_degerlendir.py _dparkktah_p24cb p24"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

APT6 = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]


def main(ek, bil):
    df = MB.load_b("train")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B, MB.DELAY]]
    df["r"] = MB.regimes(MB.load_b("train"))
    oku = lambda e: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f"BM_cv_tum{e}_oof.parquet"),  # noqa: E731
                                             on="MVT_ID_mvt", how="left")
    P, M, C = oku("_dparkktah"), oku("_dparkktahkomgspcyte_cb"), oku(ek)
    apt = df.ADEP_mvt.isin(APT6).to_numpy()
    m, m24 = apt & (df.r != 2).to_numpy(), apt
    y = df[MB.TARGET_B].to_numpy(float)
    b_l, b_c = P[f"bilesen_{bil}"].to_numpy(float), C[f"bilesen_{bil}_cb"].to_numpy(float)
    sat = {}
    for ad, v in ((f"{bil} lgb (v297)", b_l), (f"{bil} cb", b_c), (f"{bil} 0,5 / 0,5", 0.5 * b_l + 0.5 * b_c)):
        c = {k: P[f"bilesen_{k}"].to_numpy(float) for k in ("p24", "yt_tum", "d", "d24")}
        c["mu"] = 0.5 * M["bilesen_mu"].to_numpy(float) + 0.5 * M["bilesen_mu_cb"].to_numpy(float)
        c[bil] = v
        e2 = (MB.combine_tum(c, df["ADEP_mvt"].to_numpy(), yt="yt_tum", cap_non_lirf=3500.0) - y) ** 2
        r = {"6 hav. (24 sa hariç)": np.sqrt(e2[m].mean()), "6 hav. (24 sa dahil)": np.sqrt(e2[m24].mean())}
        for q, ms in V.QUARTERS.items():
            r[q] = np.sqrt(e2[m & df.MVT_MONTH.isin(ms).to_numpy()].mean())
        for a in APT6:
            r[a] = np.sqrt(e2[m & (df.ADEP_mvt == a).to_numpy()].mean())
        sat[ad] = r
    print(pd.DataFrame(sat).round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2])

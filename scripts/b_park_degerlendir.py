"""07.10 gece: Model B park yeri / komşu taksi özelliklerinin CV değerlendirmesi (b_model_cv_tum.py OOF'ları).
Ölçü 05.10'daki "durum" kararıyla aynı: üretimde tüm satırlı modelin kullanıldığı 6 havalimanı (EDDM, EGLL, LEBL, LEMD,
LFPG, LTFM), 24 saat satırları hariç, aday tum_mu_yt_tavan3500; çeyrek ve havalimanı kırılımı; ayrıca normal rejim.
Kullanım: python scripts/b_park_degerlendir.py <ek1> <ek2> ...   (ek: "" = taban, _dpark, _dparkktah)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

APT6 = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
ADAY = "tum_mu_yt_tavan3500"


def main(ekler):
    df = MB.load_b("train")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B, MB.DELAY]]
    df["r"] = MB.regimes(MB.load_b("train"))
    m = df["ADEP_mvt"].isin(APT6) & (df["r"] != 2)
    sat = {}
    for ek in ekler:
        o = pd.read_parquet(config.DATA_DIR / "preds" / f"BM_cv_tum{ek}_oof.parquet", columns=["MVT_ID_mvt", ADAY, "bilesen_mu"])
        x = df.merge(o, on="MVT_ID_mvt", how="left")
        e2 = (x[ADAY] - x[MB.TARGET_B]) ** 2
        mu2 = (x["bilesen_mu"] - x[MB.TARGET_B]) ** 2
        r = {"6 hav. (24 sa hariç)": np.sqrt(e2[m].mean()), "normal rejim μ": np.sqrt(mu2[m & (x.r == 0)].mean())}
        for q, ms in V.QUARTERS.items():
            r[q] = np.sqrt(e2[m & x.MVT_MONTH.isin(ms)].mean())
        for a in APT6:
            r[a] = np.sqrt(e2[m & (x.ADEP_mvt == a)].mean())
        sat[ek or "taban"] = r
    print(f"satır: {int(m.sum()):,} (6 havalimanı, 24 sa hariç)")
    print(pd.DataFrame(sat).round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:] or [""])

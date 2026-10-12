"""10.10: iki B5 CV rhat dosyasının karşılaştırması, güncel B tabanıyla (BM_cv_tum_guncel_oof; EDDF v126), k = 1,25 (üretim).
Kitle: üretimdeki 8 havalimanının izli B satırları (iki dosyada da düzeltilen).
Önceden tanımlı kabul (B5 değişikliği): toplam SSE iyi, ≥ 4/6 ay ve ≥ 5/8 havalimanı iyi.
Kullanım: python scripts/b5_kiyas.py ADSB_B5_cv_guncel ADSB_B5_cv_guncel_planli
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB  # noqa: E402

P = config.DATA_DIR / "preds"
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH"]
AYLAR = (1, 2, 6, 7, 8, 12)
K = 1.25


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    fa, fb = sys.argv[1], sys.argv[2]
    tr = MB.load_b("train")
    tr = tr[tr["MVT_MONTH"].isin(AYLAR)][["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B]]
    b = tr.merge(pd.read_parquet(P / "BM_cv_oof.parquet", columns=["MVT_ID_mvt", "v126_B"]), on="MVT_ID_mvt", how="left")
    b = b.merge(pd.read_parquet(P / "BM_cv_tum_guncel_oof.parquet"), on="MVT_ID_mvt", how="left")
    b["taban"] = np.where(b["ADEP_mvt"] == "EDDF", b["v126_B"], b["tum_mu_yt_tavan3500"])
    for ad, f in (("ra", fa), ("rb", fb)):
        b = b.merge(pd.read_parquet(P / f"{f}.parquet").rename(columns={"rhat": ad}), on="MVT_ID_mvt", how="inner",
                    validate="one_to_one")
    b = b[(b.ra.abs() > 1e-9) & (b.rb.abs() > 1e-9) & b.ADEP_mvt.isin(APT) & b.taban.notna()].reset_index(drop=True)
    y, h = b[MB.TARGET_B].to_numpy(float), b.taban.to_numpy(float)
    ay, apt = b.MVT_MONTH.to_numpy(), b.ADEP_mvt.to_numpy()
    print(f"izli B satırı (8 havalimanı): {len(b):,} | rhat korelasyonu {np.corrcoef(b.ra, b.rb)[0, 1]:.4f} | k = {K}")
    rows = {}
    for ad, r in (("karşılık", b.ra), ("aday", b.rb)):
        e = pd.Series((np.maximum(60.0, h + K * r.to_numpy()) - y) ** 2)
        rows[ad] = {**{f"ay {m}": np.sqrt(e[ay == m].mean()) for m in AYLAR}, **{a: np.sqrt(e[apt == a].mean()) for a in APT},
                    "toplam": np.sqrt(e.mean())}
    t = pd.DataFrame(rows).T
    t.loc["fark"] = t.loc["aday"] - t.loc["karşılık"]
    print(t.round(2).to_string())
    f = t.loc["fark"]
    ay_iyi = int(sum(f[f"ay {m}"] < 0 for m in AYLAR))
    apt_iyi = int(sum(f[a] < 0 for a in APT))
    kabul = f["toplam"] < 0 and ay_iyi >= 4 and apt_iyi >= 5
    print(f"toplam {f['toplam']:+.2f}, iyi ay {ay_iyi}/6, iyi havalimanı {apt_iyi}/8 → {'KABUL' if kabul else 'RET'}")


if __name__ == "__main__":
    main()

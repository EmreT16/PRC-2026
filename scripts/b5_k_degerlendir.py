"""10.10: B5 düzeltme katsayısı k, güncel B tabanıyla (8 havalimanı yeni B + μ'ye TabM 0,6, tavan 5.000; EDDF v126).
k = 1,25 07.10'da eski B tabanında (tum, tavan 3.500) seçilmişti; B5 A satırlarıyla eğitildiği için eğitim aynı, değişen yalnız
B satırlarının tabanı (özellik ve uygulama). CV rhat: ADSB_B5_cv_guncel (boof=_guncel, v126apt=EDDF).
Kitle: üretimdeki 8 havalimanının izli B satırları (24 sa hariç değil; B5 CV'sindeki gibi).
Önceden tanımlı kural: k ∈ {1,0; 1,5} 1,25'ten ancak toplam SSE'de iyi ve ≥ 4/6 ay ve ≥ 5/8 havalimanı iyiyse seçilir.
Kullanım: python scripts/b5_k_degerlendir.py
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
KS = [0.0, 0.75, 1.0, 1.25, 1.5, 1.75]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr = MB.load_b("train")
    tr = tr[tr["MVT_MONTH"].isin(AYLAR)][["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B]]
    oof = pd.read_parquet(P / "BM_cv_oof.parquet", columns=["MVT_ID_mvt", "v126_B"])
    tum = pd.read_parquet(P / "BM_cv_tum_guncel_oof.parquet")
    b = tr.merge(oof, on="MVT_ID_mvt", how="left").merge(tum, on="MVT_ID_mvt", how="left")
    b["taban"] = np.where(b["ADEP_mvt"] == "EDDF", b["v126_B"], b["tum_mu_yt_tavan3500"])
    b = b.merge(pd.read_parquet(P / "ADSB_B5_cv_guncel.parquet"), on="MVT_ID_mvt", how="inner", validate="one_to_one")
    b = b[(b.rhat.abs() > 1e-9) & b.ADEP_mvt.isin(APT) & b.taban.notna()].reset_index(drop=True)
    y, h, r = b[MB.TARGET_B].to_numpy(float), b.taban.to_numpy(float), b.rhat.to_numpy(float)
    ay, apt = b.MVT_MONTH.to_numpy(), b.ADEP_mvt.to_numpy()
    print(f"izli B satırı (8 havalimanı): {len(b):,}")
    S = {k: pd.Series((np.maximum(60.0, h + k * r) - y) ** 2) for k in KS}
    t = pd.DataFrame({k: {**{f"ay {m}": np.sqrt(S[k][ay == m].mean()) for m in AYLAR},
                          **{a: np.sqrt(S[k][apt == a].mean()) for a in APT}, "toplam": np.sqrt(S[k].mean())} for k in KS}).T
    print(t.round(1).to_string())
    for k in (1.0, 1.5):
        ay_iyi = int(sum(t.loc[k, f"ay {m}"] < t.loc[1.25, f"ay {m}"] for m in AYLAR))
        apt_iyi = int(sum(t.loc[k, a] < t.loc[1.25, a] for a in APT))
        sec = t.loc[k, "toplam"] < t.loc[1.25, "toplam"] and ay_iyi >= 4 and apt_iyi >= 5
        print(f"k = {k} ↔ 1,25: toplam {t.loc[k, 'toplam'] - t.loc[1.25, 'toplam']:+.2f}, iyi ay {ay_iyi}/6, iyi havalimanı "
              f"{apt_iyi}/8 → {'SEÇİLİR' if sec else 'değişmez'}")


if __name__ == "__main__":
    main()

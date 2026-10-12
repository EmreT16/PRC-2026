"""10.10 gece: B5 değişikliklerinin 12 aylık CV ile değerlendirmesi (önceden 6 ay, ~6.400 izli B satırı; kararlar gürültülüydü).
Taban: güncel B (BM_cv_tum_guncel_oof; 8 havalimanı yeni B + μ TabM, tavan 5.000; EDDF v126), k = 1,25, 8 havalimanının izli
B satırları. Karşılık: ADSB_B5_cv_g12 (LightGBM, üretim ayarları).
  nn=<dosya>: rhat = (1 − v) · LightGBM + v · aday; iç içe ay dışarıda v seçimi.
  aday=<dosya>: doğrudan karşılaştırma (ör. planli).
Önceden tanımlı kabul: (iç içe seçimle) toplam iyi, (seçilen v'de) ≥ 8/12 ay ve ≥ 5/8 havalimanı iyi.
Kullanım: python scripts/b5_degerlendir12.py nn=ADSB_B5_cv_g12nn  |  aday=ADSB_B5_cv_g12pl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB  # noqa: E402

P = config.DATA_DIR / "preds"
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH"]
K = 1.25


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    f2 = opt.get("nn") or opt["aday"]
    tr = MB.load_b("train")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B]]
    b = tr.merge(pd.read_parquet(P / "BM_cv_oof.parquet", columns=["MVT_ID_mvt", "v126_B"]), on="MVT_ID_mvt", how="left")
    b = b.merge(pd.read_parquet(P / "BM_cv_tum_guncel_oof.parquet"), on="MVT_ID_mvt", how="left")
    b["taban"] = np.where(b["ADEP_mvt"] == "EDDF", b["v126_B"], b["tum_mu_yt_tavan3500"])
    for ad, f in (("rl", opt.get("lgb", "ADSB_B5_cv_g12")), ("rn", f2)):
        b = b.merge(pd.read_parquet(P / f"{f}.parquet").rename(columns={"rhat": ad}), on="MVT_ID_mvt", how="inner",
                    validate="one_to_one")
    b = b[(b.rl.abs() > 1e-9) & (b.rn.abs() > 1e-9) & b.ADEP_mvt.isin(APT) & b.taban.notna()].reset_index(drop=True)
    y, h = b[MB.TARGET_B].to_numpy(float), b.taban.to_numpy(float)
    ay, apt = b.MVT_MONTH.to_numpy(), b.ADEP_mvt.to_numpy()
    aylar = sorted(set(ay.tolist()))
    VS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] if "nn" in opt else [0.0, 1.0]
    print(f"izli B satırı (8 havalimanı, {len(aylar)} ay): {len(b):,} | rhat korelasyonu {np.corrcoef(b.rl, b.rn)[0, 1]:.4f} | k = {K}")
    E = {v: (np.maximum(60.0, h + K * ((1 - v) * b.rl.to_numpy() + v * b.rn.to_numpy())) - y) ** 2 for v in VS}
    t = pd.DataFrame({v: {**{f"ay {m}": np.sqrt(E[v][ay == m].mean()) for m in aylar}, **{a: np.sqrt(E[v][apt == a].mean()) for a in APT},
                          "toplam": np.sqrt(E[v].mean())} for v in VS}).T
    print(t.round(1).to_string())
    ic = 0.0
    sec = {}
    for m in aylar:
        sec[m] = min(VS[1:], key=lambda v: E[v][ay != m].sum())
        ic += E[sec[m]][ay == m].sum()
    ic_f = np.sqrt(ic / len(b)) - t.loc[0.0, "toplam"]
    en = min(VS[1:], key=lambda v: E[v].sum())
    ay_iyi = int(sum(t.loc[en, f"ay {m}"] < t.loc[0.0, f"ay {m}"] for m in aylar))
    apt_iyi = int(sum(t.loc[en, a] < t.loc[0.0, a] for a in APT))
    kabul = ic_f < 0 and ay_iyi >= 8 and apt_iyi >= 5
    print(f"iç içe seçimler {sec} → toplam {ic_f:+.2f} | tüm aylarla v = {en} ({t.loc[en, 'toplam'] - t.loc[0.0, 'toplam']:+.2f}), "
          f"iyi ay {ay_iyi}/{len(aylar)}, iyi havalimanı {apt_iyi}/8 → {'KABUL' if kabul else 'RET'}")


if __name__ == "__main__":
    main()

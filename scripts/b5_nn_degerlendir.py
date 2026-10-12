"""10.10: Model B düzelticisinde (B5) LightGBM + TabM harmanı (CV; A satırlarıyla eğitim, B satırlarında sınama, 6 ay).
rhat = (1 − v) · rhat_LightGBM (ADSB_B5_cv_lgb10) + v · rhat_TabM (ADSB_B5_cv_nn10); taksi = max(60, taban + k · rhat), k = 1,25
(üretimdeki). Kitle: üretimdeki 8 havalimanının izli B satırları (LIRF / LTFM B5'te düzeltilmiyor).
Önceden tanımlı kabul: (1) iç içe ay dışarıda seçim (her ay için v kalan 5 aydan) toplamda v = 0'dan iyi, (2) tüm aylarla seçilen
v'de ≥ 4/6 ay ve ≥ 5/8 havalimanı iyi.
Kullanım: python scripts/b5_nn_degerlendir.py [boof=_dparkktah] [k=1.25]
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
VS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    k = float(opt.get("k", 1.25))
    tr = MB.load_b("train")
    tr = tr[tr["MVT_MONTH"].isin(AYLAR)][["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B]]
    oof = pd.read_parquet(P / "BM_cv_oof.parquet", columns=["MVT_ID_mvt", "v126_B"])
    tum = pd.read_parquet(P / f"BM_cv_tum{opt.get('boof', '_dparkktah')}_oof.parquet", columns=["MVT_ID_mvt", "tum_mu_yt_tavan3500"])
    b = tr.merge(oof, on="MVT_ID_mvt", how="left").merge(tum, on="MVT_ID_mvt", how="left")
    b["taban"] = np.where(b["ADEP_mvt"].isin(["EDDF"]), b["v126_B"], b["tum_mu_yt_tavan3500"])
    for ad, f in (("rl", "ADSB_B5_cv_lgb10"), ("rn", "ADSB_B5_cv_nn10")):
        b = b.merge(pd.read_parquet(P / f"{f}.parquet").rename(columns={"rhat": ad}), on="MVT_ID_mvt", how="inner",
                    validate="one_to_one")
    iz = (b.rl.abs() > 1e-9) & (b.rn.abs() > 1e-9) & b.ADEP_mvt.isin(APT) & b.taban.notna()
    b = b[iz].reset_index(drop=True)
    y, h = b[MB.TARGET_B].to_numpy(float), b.taban.to_numpy(float)
    ay, apt = b.MVT_MONTH.to_numpy(), b.ADEP_mvt.to_numpy()
    print(f"izli B satırı (8 havalimanı): {len(b):,} | rhat korelasyonu {np.corrcoef(b.rl, b.rn)[0, 1]:.4f} | k = {k}")
    sse = {}
    for v in VS:
        p = np.maximum(60.0, h + k * ((1 - v) * b.rl.to_numpy() + v * b.rn.to_numpy()))
        sse[v] = pd.Series((p - y) ** 2).groupby([apt, ay]).sum()
    S = pd.DataFrame(sse)
    S.index.names = ["apt", "ay"]
    N = pd.Series(np.ones(len(b))).groupby([apt, ay]).sum()
    taban_sse = pd.Series((np.maximum(60.0, h) - y) ** 2).groupby(ay).sum()
    t = pd.DataFrame({v: np.sqrt(S[v].groupby("ay").sum() / N.groupby(level=1).sum()) for v in VS}).T
    t["toplam"] = [np.sqrt(S[v].sum() / N.sum()) for v in VS]
    ta = pd.DataFrame({v: np.sqrt(S[v].groupby("apt").sum() / N.groupby(level=0).sum()) for v in VS}).T
    print("düzelticisiz taban (toplam):", round(float(np.sqrt(taban_sse.sum() / N.sum())), 2))
    print(pd.concat([t, ta], axis=1).round(2).to_string())
    # iç içe ay dışarıda
    ic = 0.0
    sec = {}
    for m in AYLAR:
        egit = S[S.index.get_level_values("ay") != m].sum()
        sec[m] = egit.idxmin()
        ic += S[S.index.get_level_values("ay") == m][sec[m]].sum()
    ic_f = np.sqrt(ic / N.sum()) - t.loc[0.0, "toplam"]
    en = t["toplam"].idxmin()
    ay_iyi = int((t.loc[en, list(AYLAR)] < t.loc[0.0, list(AYLAR)]).sum())
    apt_iyi = int((ta.loc[en] < ta.loc[0.0]).sum())
    kabul = ic_f < 0 and ay_iyi >= 4 and apt_iyi >= 5
    print(f"iç içe seçimler {sec} → toplam {ic_f:+.3f} | tüm aylarla en iyi v = {en} ({t.loc[en, 'toplam'] - t.loc[0.0, 'toplam']:+.3f}), "
          f"iyi ay {ay_iyi}/6, iyi havalimanı {apt_iyi}/8 → {'KABUL' if kabul else 'RET'}")


if __name__ == "__main__":
    main()

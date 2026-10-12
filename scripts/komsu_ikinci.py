"""Komşu ikinci aşaması (02.10 gece). Taban: PRC_HARMAN harmanının OOF ofseti (src/adsb.py
d_dw_split). Kalıntı (taksi uzayında) ~ [apt, harman] + _kqgp paketi + komşu kalkış aralıkları (data/komsu_oz.parquet,
scripts/komsu_ozellik.py) + komşuların TAHMİN edilen taksisi (önceki/sonraki 10 kalkış ve aynı pistte 5; kendisi hariç).
Tarama (scripts/tarama_olay.py ay; kontrol = aynı paket): lgbh3 üstünde −0,75 (6/6 ay, 10/10 havalimanı).
Çıktı (PRC_IKINCI=<ad> düzeni, src/adsb.py): data/preds/IKINCI_cv_<ad>.parquet (ay dışı OOF) ve TEST_IKINCI_<ad>.parquet.
Kullanım: PRC_HARMAN=kqgpxg python scripts/komsu_ikinci.py <ad> [split_1,split_2,split_3[,split_4]] [cv|hepsi]
"""
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PRC_GECPIST"] = "1"
import adsb as A  # noqa: E402
from src import config  # noqa: E402
from src import features_v126 as F  # noqa: E402

PRM = dict(objective="huber", alpha=600.0, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=1, num_threads=8)
TUR = 300
DOSYA = ("gecikme_oz", "gecikme_pist_oz", "varis_wx_oz", "pencere_ileri_oz", "park_varis_oz", "komsu_oz")
KOM = [f"KOM_{t}_{k}" for t in ("PP", "PS") for k in (2, 3, 5, 10)] + [f"KOM_{t}_{k}" for t in ("AP", "AS") for k in (5, 10)]
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
KTAH = ["KTAH_ORT", "KTAH_FARK", "KTAH_P_ORT", "KTAH_P_FARK"]


def komsu_ort(v, k):
    cs, n = np.concatenate([[0.0], np.cumsum(v)]), len(v)
    i = np.arange(n)
    lo, hi = np.maximum(i - k, 0), np.minimum(i + k + 1, n)
    c = hi - lo - 1
    return np.where(c > 0, (cs[hi] - cs[lo] - v) / np.maximum(c, 1), np.nan)


def hazirla(d):
    """d: MVT_ID_mvt, ADEP_mvt, NM_TAXI_DIFF_SEC, harman (ofset) → özellik çerçevesi + taban taksi."""
    for f in DOSYA:
        d = d.merge(pd.read_parquet(config.DATA_DIR / f"{f}.parquet"), on="MVT_ID_mvt", how="left", validate="one_to_one")
    d["taban"] = np.maximum(60, d["NM_TAXI_DIFF_SEC"].to_numpy(float) - d["harman"].to_numpy(float))
    tah, Tk = d["taban"].to_numpy(float), d["KOM_T"].to_numpy(float)
    for ad, anah, k in (("KTAH", ["ADEP_mvt"], 10), ("KTAH_P", ["ADEP_mvt", "KOM_PIST"], 5)):
        ort = np.full(len(d), np.nan)
        for _, idx in d.groupby(anah).indices.items():
            o = idx[np.argsort(Tk[idx], kind="stable")]
            ort[o] = komsu_ort(tah[o], k)
        d[f"{ad}_ORT"], d[f"{ad}_FARK"] = ort, tah - ort
    d["apt"] = pd.Categorical(d["ADEP_mvt"].astype(str).to_numpy(), categories=APT)
    return d


def kolonlar():
    if os.environ.get("KOMSU_SADE") == "1":  # yalnız [apt, harman] + komşu (harmanın kendi özellikleri yeniden öğrenilmez)
        return ["apt", "harman"] + KOM + KTAH
    gec = [c for c in pd.read_parquet(config.DATA_DIR / "gecikme_oz.parquet").columns if c != "MVT_ID_mvt"]
    return (["apt", "harman"] + gec + F.GEC_PIST_COLS + F.VARIS_WX_COLS + F.ILERI_COLS + F.PARK_VARIS_COLS + KOM + KTAH)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    assert not A.IKINCI, "PRC_IKINCI kapalıyken çalıştırılmalı"
    ad = sys.argv[1]
    splits = (sys.argv[2] if len(sys.argv) > 2 else "split_1,split_2,split_3").split(",")
    kip = sys.argv[3] if len(sys.argv) > 3 else "hepsi"
    t0 = time.time()
    d = pd.concat([A.d_dw_split(s) for s in splits], ignore_index=True).rename(columns={"ofset_taban": "harman"})
    d = hazirla(d)
    cols = kolonlar()
    yt = (d["NM_TAXI_DIFF_SEC"] - d["y"]).to_numpy(float)
    r = yt - d["taban"].to_numpy(float)
    ay = d["MVT_MONTH"].to_numpy()
    rhat = np.zeros(len(d))
    for m in sorted(set(ay)):
        tr, va = ay != m, ay == m
        rhat[va] = lgb.train(PRM, lgb.Dataset(d.loc[tr, cols], r[tr]), TUR).predict(d.loc[va, cols])
    haric = [a for a in os.environ.get("KOMSU_HARIC", "").split(",") if a]  # ör. LIRF: düzeltme bu havalimanlarında 0
    rhat[d["ADEP_mvt"].isin(haric).to_numpy()] = 0.0
    p0, p1 = d["taban"].to_numpy(float), np.maximum(60, d["taban"].to_numpy(float) + rhat)
    rm = lambda p, m: np.sqrt(np.mean((yt[m] - p[m]) ** 2))  # noqa: E731
    fark = [rm(p1, ay == m) - rm(p0, ay == m) for m in sorted(set(ay))]
    print("ay: " + " ".join(f"{m}: {rm(p0, ay == m):.2f} → {rm(p1, ay == m):.2f}" for m in sorted(set(ay)))
          + f" | ort {np.mean(fark):+.3f} | iyi {sum(f < 0 for f in fark)}/{len(fark)}", flush=True)
    A_ = d["ADEP_mvt"].to_numpy()
    print("havalimanı: " + " ".join(f"{a} {rm(p1, A_ == a) - rm(p0, A_ == a):+.2f}" for a in sorted(set(A_))), flush=True)
    pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "rhat": rhat}).to_parquet(config.DATA_DIR / "preds" / f"IKINCI_cv_{ad}.parquet",
                                                                          index=False)
    if kip == "hepsi":
        t = pd.read_parquet(A.PRED / f"TEST_{A.TEST_AD}.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "harman"])
        t = t.merge(pd.read_parquet(config.DATA_DIR / "base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]),
                    on="MVT_ID_mvt", how="left", validate="one_to_one")
        t = hazirla(t[t["NM_TAXI_DIFF_SEC"].notna()].reset_index(drop=True))
        X, Xt = d[cols], t[cols]
        rt = np.mean([lgb.train(dict(PRM, seed=s), lgb.Dataset(X, r), TUR).predict(Xt) for s in (1, 2026, 7)], axis=0)
        rt[t["ADEP_mvt"].isin(haric).to_numpy()] = 0.0
        pd.DataFrame({"MVT_ID_mvt": t["MVT_ID_mvt"], "rhat": rt}).to_parquet(
            config.DATA_DIR / "preds" / f"TEST_IKINCI_{ad}.parquet", index=False)
        print(f"test {len(t):,} | rhat ort {rt.mean():.1f}, std {rt.std():.1f} | OOF rhat ort {rhat.mean():.1f}, std {rhat.std():.1f}")
    print(f"süre {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()

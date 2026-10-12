"""Aday özellik taraması (02.10 akşam). lgbh3_kqgp 6 ay OOF kalıntısına küçük LGB.
kontrol = [apt, harman] + tam _kqgp paketi (gecikme, pist, varış+METAR, ileri pencere, park varışı); aday = kontrol + grup.
Kip:
  ay  : ay dışarıda (6 kat). Çıktı: aylık taksi RMSE farkı (aday − kontrol), havalimanı ortalaması.
  kar : kar olayı dışarıda. Kar günü = havalimanının o UTC gününde METAR'da ≥ 3 saat kar (kar_oz.KAR_GUN). Kar tarihleri
        art arda günlerle kümelenir, kümeler 5 kata dağıtılır; karsız günler gün sırasına göre aynı 5 kata. Çıktı: kar
        satırlarında ve karsız satırlarda RMSE farkı, kar satırlarında havalimanı bazında.
Kullanım: python scripts/tarama_olay.py <ay|kar> [GRUP1,GRUP2,...]
"""
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PRC_GECPIST"] = "1"
from src import config  # noqa: E402
from src import features_v126 as F  # noqa: E402

P = config.DATA_DIR / "preds"
BIL = os.environ.get("TARAMA_BIL", "ADAY_v_lgb_huber1000_3t_kqgp")
KIP = sys.argv[1]
K2 = os.environ.get("TARAMA_K2") == "1"  # 03.10 akşam: komşu 2. tur (scripts/komsu2_oz.py); taban kqgpy, kontrol + KOM + KTAH
d = pd.concat([pd.read_parquet(P / f"B1_{s}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "MVT_DAY",
                                                               "NM_TAXI_DIFF_SEC", "y"]).merge(
    pd.read_parquet(P / f"{BIL}_{s}.parquet", columns=["MVT_ID_mvt", "harman"]), on="MVT_ID_mvt", validate="one_to_one")
    for s in ("split_1", "split_2", "split_3")], ignore_index=True)
for f in ("gecikme_oz", "gecikme_pist_oz", "varis_wx_oz", "pencere_ileri_oz", "park_varis_oz", "yuzey_oz", "kar_oz", "komsu_oz", "yanki_te_oz", "pist_oz", "gunluk_oz"):
    if (config.DATA_DIR / f"{f}.parquet").exists():
        d = d.merge(pd.read_parquet(config.DATA_DIR / f"{f}.parquet"), on="MVT_ID_mvt", how="left", validate="one_to_one")
rng = np.random.default_rng(0)
for k in range(3):
    d[f"GURULTU_{k}"] = rng.normal(size=len(d)).astype("float32")
d["apt"] = pd.Categorical(d["ADEP_mvt"].astype(str).to_numpy())
nm, y, p = d["NM_TAXI_DIFF_SEC"].to_numpy(float), d["y"].to_numpy(float), d["harman"].to_numpy(float)
if K2:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from komsu2_oz import GRUPLAR as K2G, ozellik
    d = d.merge(pd.read_parquet(config.DATA_DIR / "komsu_tah_oz.parquet"), on="MVT_ID_mvt", how="left", validate="one_to_one")
    x = d[["ADEP_mvt", "KOM_T", "KOM_PIST"]].assign(nm=nm, tah=np.maximum(60, nm - p))
    d = pd.concat([d, ozellik(x)], axis=1)
elif "KOM_T" in d:  # 02.10 gece: komşu kalkışların TAHMİN edilen taksisi (önceki/sonraki k), kendisi hariç
    tah, Tk = np.maximum(60, nm - p), d["KOM_T"].to_numpy(float)

    def komsu_ort(v, k):
        cs, n = np.concatenate([[0.0], np.cumsum(v)]), len(v)
        i = np.arange(n); lo, hi = np.maximum(i - k, 0), np.minimum(i + k + 1, n)
        c = hi - lo - 1
        return np.where(c > 0, (cs[hi] - cs[lo] - v) / np.maximum(c, 1), np.nan)

    for ad, anah, k in (("KTAH", ["ADEP_mvt"], 10), ("KTAH_P", ["ADEP_mvt", "KOM_PIST"], 5)):
        ort = np.full(len(d), np.nan)
        for _, idx in d.groupby(anah).indices.items():
            o = idx[np.argsort(Tk[idx], kind="stable")]
            ort[o] = komsu_ort(tah[o], k)
        d[f"{ad}_ORT"], d[f"{ad}_FARK"] = ort, tah - ort
r = y - p
gec = [c for c in pd.read_parquet(config.DATA_DIR / "gecikme_oz.parquet").columns if c != "MVT_ID_mvt"]
KON = ["apt", "harman"] + gec + F.GEC_PIST_COLS + F.VARIS_WX_COLS + F.ILERI_COLS + F.PARK_VARIS_COLS
GRUP = {"NOISE": ["GURULTU_0", "GURULTU_1", "GURULTU_2"],
        "YZ": ["ARR_TI_W30_MED", "ARR_TI_W30_MAX", "ARR_TI_W30_N", "ARR_TI_W15_MED", "ARR_TI_F15_MED"],
        "NMG": ["NMT_C15_G", "NMT_F15_G", "NMT_PIST_C15_G", "NMT_YAPAY"],
        "KAR": ["KAR_3S", "KAR_6S", "KAR_12S", "KAR_24S", "DON_6S", "KAR_SONRA_3S"],
        "KARG": ["KAR_GUN", "KAR_ONCEKI_GUN"],
        "SIC": ["SIC_0", "SIC_MIN12", "SPREAD_0"]}
GRUP["YZNMG"] = GRUP["YZ"] + GRUP["NMG"]
GRUP["KOM"] = [f"KOM_{t}_{k}" for t in ("PP", "PS") for k in (2, 3, 5, 10)] + [f"KOM_{t}_{k}" for t in ("AP", "AS") for k in (5, 10)]
GRUP["KTAH"] = ["KTAH_ORT", "KTAH_FARK", "KTAH_P_ORT", "KTAH_P_FARK"]
GRUP["KOMTAH"] = GRUP["KOM"] + GRUP["KTAH"]
GRUP["YTE"] = ["YTE_UCUS", "YTE_HY", "YTE_VARIS", "YTE_PARK", "YTE_N_UCUS"]  # 04.10: yankı geçmişi (yanki_te_ozellik.py)
GRUP["YTE2"] = ["YTE_UCUS", "YTE_HY"]
GRUP["PIST"] = ["PC_N_DEP", "PC_N_ARR", "PC_DEGISTI", "PC_PAY", "PC_BASKIN", "PC_KUYRUK_W", "PC_AGIR_N", "PC_SPAN20", "PC_CFG_TE"]  # 04.10
GRUP["GUN"] = ["GS_REG_PAY", "GS_GEC_PAY", "GS_ERK_PAY", "GS_ONCE_DK", "GS_ATC_DK"]  # 04.10 EUROCONTROL günlük
GRUP["PISTK"] = ["PC_KUYRUK_W", "PC_AGIR_N", "PC_SPAN20"]
GRUP["PISTC"] = ["PC_N_DEP", "PC_N_ARR", "PC_DEGISTI", "PC_PAY", "PC_BASKIN", "PC_CFG_TE"]
GRUP["KARHEP"] = GRUP["KAR"] + GRUP["KARG"]
GRUP["KARW"] = GRUP["KARHEP"]  # + kar günü satırlarına 3 kat ağırlık
if K2:
    KON = KON + GRUP["KOMTAH"]
    GRUP = {"NOISE": GRUP["NOISE"], **K2G, "K2HEP": [c for g in K2G.values() for c in g]}
if len(sys.argv) > 2:
    GRUP = {k: GRUP[k] for k in sys.argv[2].split(",")}
prm = dict(objective="huber", alpha=600.0, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=1, num_threads=8)
ay = d["MVT_MONTH"].to_numpy()
kar = (d["KAR_GUN"].fillna(0).to_numpy() >= 3)
tarih = pd.to_datetime(dict(year=2025, month=d["MVT_MONTH"], day=d["MVT_DAY"]))
if KIP == "ay":
    kat = ay.copy()
else:
    kt = np.sort(tarih[kar].unique())
    kume, k = {}, -1
    for i, t in enumerate(kt):
        if i == 0 or (t - kt[i - 1]) > np.timedelta64(1, "D"):
            k += 1
        kume[t] = k
    gs = tarih.dt.dayofyear.to_numpy()
    kat = np.where(tarih.isin(kume.keys()), tarih.map(kume).fillna(-1).to_numpy() % 5, gs % 5).astype(int)
    print(f"kar satırı {kar.sum():,} ({len(kt)} tarih, {k + 1} küme); havalimanı: "
          + str(d.loc[kar, "ADEP_mvt"].value_counts().to_dict()), flush=True)
KATLAR = sorted(set(kat))


def rm(pp, m):
    return np.sqrt(np.mean((np.maximum(60, nm[m] - pp[m]) - (nm[m] - y[m])) ** 2)) if m.any() else np.nan


def oof(cols, w=None):
    out = np.zeros(len(d))
    for m in KATLAR:
        tr, va = kat != m, kat == m
        ds = lgb.Dataset(d.loc[tr, cols], r[tr], weight=None if w is None else w[tr])
        out[va] = lgb.train(prm, ds, 300).predict(d.loc[va, cols])
    return p + out


kon = oof(KON)
apts = sorted(d["ADEP_mvt"].unique())
A = d["ADEP_mvt"].to_numpy()
if KIP == "ay":
    print("kontrol:", " ".join(f"{m}:{rm(kon, ay == m):.2f}" for m in KATLAR), flush=True)
else:
    print(f"kontrol: kar {rm(kon, kar):.2f} | karsız {rm(kon, ~kar):.2f} | kar hv: "
          + " ".join(f"{a} {rm(kon, kar & (A == a)):.1f}" for a in apts if (kar & (A == a)).any()), flush=True)
for g, cols in GRUP.items():
    w = np.where(kar, 3.0, 1.0) if g == "KARW" else None
    q = oof(KON + cols, w)
    if KIP == "ay":
        fark = [rm(q, ay == m) - rm(kon, ay == m) for m in KATLAR]
        hv = {a: np.mean([rm(q, (A == a) & (ay == m)) - rm(kon, (A == a) & (ay == m)) for m in KATLAR]) for a in apts}
        print(f"{g}: " + " ".join(f"{m}:{f:+.2f}" for m, f in zip(KATLAR, fark)) + f" | ort {np.mean(fark):+.3f} | iyi "
              f"{sum(f < 0 for f in fark)}/6 | hv: " + " ".join(f"{a} {v:+.2f}" for a, v in hv.items()), flush=True)
    else:
        kf = [rm(q, kar & (kat == m)) - rm(kon, kar & (kat == m)) for m in KATLAR]
        print(f"{g}: kar {rm(q, kar) - rm(kon, kar):+.2f} (katlar " + " ".join(f"{f:+.1f}" for f in kf)
              + f") | karsız {rm(q, ~kar) - rm(kon, ~kar):+.3f} | genel {rm(q, kar | ~kar) - rm(kon, kar | ~kar):+.3f} | kar hv: "
              + " ".join(f"{a} {rm(q, kar & (A == a)) - rm(kon, kar & (A == a)):+.1f}" for a in apts if (kar & (A == a)).any()),
              flush=True)

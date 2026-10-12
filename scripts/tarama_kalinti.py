"""Hızlı tarama (01.10 gece): lgbh3_kqg OOF kalıntısı üzerinde ay dışı küçük LGB. kontrol = [apt, harman, GEC_*, GEC_PIST_*];
her grup kontrol + grup. Çıktı: aylık (grup − kontrol) taksi RMSE farkı."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
SC = config.DATA_DIR
P = config.DATA_DIR / "preds"
BIL = sys.argv[1] if len(sys.argv) > 1 else "ADAY_v_lgb_huber1000_3t_kqg"
d = pd.concat([pd.read_parquet(P / f"B1_{s}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y"]).merge(
    pd.read_parquet(P / f"{BIL}_{s}.parquet", columns=["MVT_ID_mvt", "harman"]), on="MVT_ID_mvt", validate="one_to_one")
    for s in ("split_1", "split_2", "split_3")], ignore_index=True)
for f in (config.DATA_DIR / "gecikme_oz.parquet", config.DATA_DIR / "gecikme_pist_oz.parquet", SC / "aday_oz.parquet", SC / "aday_oz2.parquet", config.DATA_DIR / "wx_oz.parquet"):
    d = d.merge(pd.read_parquet(f), on="MVT_ID_mvt", how="left", validate="one_to_one")
rng = np.random.default_rng(0)
for k in range(3): d[f"GURULTU_{k}"] = rng.normal(size=len(d)).astype("float32")
d["apt"] = pd.Categorical(d["ADEP_mvt"].astype(str).to_numpy())
nm, y, p = d["NM_TAXI_DIFF_SEC"].to_numpy(float), d["y"].to_numpy(float), d["harman"].to_numpy(float)
r = y - p
GEC = ["GEC_EOBT_G15", "GEC_EOBT_G30", "GEC_SCHED_G60", "GEC_N_G15", "GEC_KENDI_FARK",
       "GEC_PIST_EOBT_G15", "GEC_PIST_EOBT_G30", "GEC_PIST_N_G15", "GEC_PIST_FARK_G15"]
GRUP = {"PD": ["PD_PAY_60", "PD_DIGER_SON", "PD_PIST_YAS"], "AR": ["AR_PIST_15", "AR_APT_15", "AR_PIST_PAY_60"],
        "VG": ["VAR_GEC_G30"], "PS": ["GEC_PIST_SCHED_G60"],
        "WX": ["WX_FZ_0", "WX_FG_0", "WX_SN_0", "WX_SN_3S", "WX_TS_0", "WX_GORUS", "WX_RUZGAR", "WX_HAMLE"]}
GRUP["AL"] = ["AL_GEC_G60", "AL_N_G60"]
GRUP["APR"] = ["APR_GEC_G30", "APR_DEP_20", "APR_ARR_20"]
GRUP["GO"] = ["GUN_ORAN"]
GRUP["NMY"] = ["NM_YAPAY", "NM_YAPAY_MERKEZ"]
GRUP["NOISE"] = ["GURULTU_0", "GURULTU_1", "GURULTU_2"]
GRUP["ARWX"] = GRUP["AR"] + GRUP["WX"]
GRUP["WXF"] = ["WX_FZ_0", "WX_FG_0", "WX_SN_0", "WX_SN_3S", "WX_TS_0", "WX_GORUS"]
TUM = dict(GRUP)
if len(sys.argv) > 2: GRUP = {k: GRUP[k] for k in sys.argv[2].split(",")}
prm = dict(objective="huber", alpha=600.0, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=1, num_threads=4)
ay = d["MVT_MONTH"].to_numpy(); AY = sorted(set(ay))
def rm(pp, m): return np.sqrt(np.mean((np.maximum(60, nm[m] - pp[m]) - (nm[m] - y[m])) ** 2))
def oof(cols):
    out = np.zeros(len(d))
    for m in AY:
        tr, va = ay != m, ay == m
        out[va] = lgb.train(prm, lgb.Dataset(d.loc[tr, cols], r[tr]), 300).predict(d.loc[va, cols])
    return p + out
EKK = sum((TUM[k] for k in sys.argv[3].split(",")), []) if len(sys.argv) > 3 else []
GEC = GEC + EKK
kon = oof(["apt", "harman"] + GEC)
print("kontrol:", " ".join(f"{m}:{rm(kon, ay == m):.2f}" for m in AY), flush=True)
for g, cols in GRUP.items():
    q = oof(["apt", "harman"] + GEC + cols)
    fark = [rm(q, ay == m) - rm(kon, ay == m) for m in AY]
    apt = {a: np.mean([rm(q, (d["ADEP_mvt"] == a).to_numpy() & (ay == m)) - rm(kon, (d["ADEP_mvt"] == a).to_numpy() & (ay == m)) for m in AY])
           for a in sorted(d["ADEP_mvt"].unique())}
    print(f"{g}: " + " ".join(f"{m}:{f:+.2f}" for m, f in zip(AY, fark)) + f" | ort {np.mean(fark):+.3f} | iyi {sum(f < 0 for f in fark)}/6"
          + " | hv: " + " ".join(f"{a} {v:+.2f}" for a, v in apt.items()), flush=True)

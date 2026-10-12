"""Tarama için aday özellikler (01.10 gece, scratch): pist değişimi, pist bazlı varış yoğunluğu, varış gecikmesi.
Etiket yok; testte de hesaplanabilir. Çıktı: data/aday_oz.parquet"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K
from gecikme_ozellik import pencere_medyan
KOL = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "SCHED_TIME_UTC_mvt"]
CIK = Path(__file__).resolve().parents[1] / "data" / "aday_oz.parquet"

def say(s, a, b):  # a <= x < b
    return np.searchsorted(s, b, "left") - np.searchsorted(s, a, "left")

def apt_blok(dep, arr):
    dep = dep.sort_values("T", kind="stable"); T = dep["T"].to_numpy(float); R = dep["pist"].to_numpy()
    n = len(dep); out = {k: np.full(n, np.nan) for k in ["PD_PAY_60", "PD_DIGER_SON", "PD_PIST_YAS", "AR_PIST_15", "AR_APT_15", "AR_PIST_PAY_60", "VAR_GEC_G30", "GEC_PIST_SCHED_G60"]}
    Ts = T
    tum60 = say(Ts, T - 3600, T)
    At = np.sort(arr["T"].to_numpy(float))
    out["AR_APT_15"] = say(At, T - 900, T).astype(float)
    a60 = say(At, T - 3600, T)
    ao = arr.sort_values("T", kind="stable")
    # varış gecikmesi ortancası: kalkış anından önceki 30 dk'daki varışlar
    if len(ao):
        aT, aG = ao["T"].to_numpy(float), ao["g"].to_numpy(float)
        lo, hi = np.searchsorted(aT, T - 1800, "left"), np.searchsorted(aT, T, "left")
        out["VAR_GEC_G30"] = np.array([np.nanmedian(aG[l:h]) if h > l else np.nan for l, h in zip(lo, hi)])
    for r in np.unique(R):
        i = np.where(R == r)[0]; t = T[i]
        out["PD_PAY_60"][i] = np.where(tum60[i] > 0, say(t, t - 3600, t) / np.maximum(tum60[i], 1), np.nan)
        diger = np.sort(T[R != r])
        j = np.searchsorted(diger, t, "left")
        out["PD_DIGER_SON"][i] = np.where(j > 0, t - diger[np.clip(j - 1, 0, None)] if len(diger) else np.nan, np.nan) if len(diger) else np.nan
        # oturum başlangıcı: aynı pistte önceki kalkışla arası > 45 dk olan ilk kalkış
        bas = np.empty(len(t)); cur = t[0] if len(t) else 0
        for k in range(len(t)):
            if k == 0 or t[k] - t[k - 1] > 2700: cur = t[k]
            bas[k] = cur
        out["PD_PIST_YAS"][i] = t - bas
        ar = np.sort(arr[arr["pist"] == r]["T"].to_numpy(float))
        out["AR_PIST_15"][i] = say(ar, t - 900, t)
        out["AR_PIST_PAY_60"][i] = np.where(a60[i] > 0, say(ar, t - 3600, t) / np.maximum(a60[i], 1), np.nan)
        s = dep["gs"].to_numpy(float)[i]
        out["GEC_PIST_SCHED_G60"][i], _ = pencere_medyan(t, s, 3600.0)
    for k in ("PD_DIGER_SON", "PD_PIST_YAS"): out[k] = np.minimum(out[k], 21600.0)
    return pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy(), **out})

def blok(df):
    df = df.copy(); df["T"] = K.ep(df["MVT_TIME_UTC_mvt"]); df["pist"] = df["RUNWAY_mvt"].astype(str)
    df["gs"] = np.clip(df["T"] - K.ep(df["SCHED_TIME_UTC_mvt"]), -3600, 14400)
    df["g"] = df["gs"]
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")]
    return pd.concat([apt_blok(d, arr[arr["ADES_mvt"].eq(a)]) for a, d in dep.groupby("ADEP_mvt")], ignore_index=True)

t0 = time.time()
tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)], ignore_index=True)
rk = pd.read_parquet(K.config.RANK_RAW, columns=KOL)
out = pd.concat([blok(tr), blok(rk)], ignore_index=True)
assert out["MVT_ID_mvt"].is_unique
out.astype({c: "float32" for c in out.columns[1:]}).to_parquet(CIK, index=False)
print(f"{len(out):,} | {time.time() - t0:.0f} sn"); print(out.describe().T.round(2).to_string())

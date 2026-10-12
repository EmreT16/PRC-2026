"""Tarama adayları 2 (01.10 gece, scratch): havayolu gecikme endeksi, apron (park grubu) yoğunluğu / gecikmesi.
Park grubu = STAND_mvt'nin harf öneki + ilk rakamı (kaba). Varışların park anı ≈ iniş + 480 sn.
Çıktı: data/aday_oz2.parquet"""
import sys, time, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K
from gecikme_ozellik import pencere_medyan
KOL = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt", "FLIGHT_mvt", "EOBT_1_flt", "AOBT_3_flt"]
CIK = Path(__file__).resolve().parents[1] / "data" / "aday_oz2.parquet"

def grup(s):
    s = s.fillna("").astype(str).str.upper().str.strip()
    return s.str.extract(r"^([A-Z]*\d?)", expand=False).fillna("")

def say(s, a, b): return np.searchsorted(s, b, "left") - np.searchsorted(s, a, "left")

def apt_blok(dep, arr):
    dep = dep.sort_values("T", kind="stable")
    n = len(dep); out = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy()})
    for k in ("AL_GEC_G60", "AL_N_G60", "APR_GEC_G30", "APR_DEP_20", "APR_ARR_20"): out[k] = np.nan
    T, O, G = dep["T"].to_numpy(float), dep["O"].to_numpy(float), dep["g"].to_numpy(float)
    for _, i in dep.groupby("al").indices.items():
        m, c = pencere_medyan(T[i], G[i], 3600.0)
        out.loc[i, "AL_GEC_G60"], out.loc[i, "AL_N_G60"] = m, c
    arr_g = {k: np.sort(v["P"].to_numpy(float)) for k, v in arr.groupby("pg")}
    for k, i in dep.groupby("pg").indices.items():
        if k == "":
            continue
        m, _ = pencere_medyan(T[i], G[i], 1800.0)
        out.loc[i, "APR_GEC_G30"] = m
        o = O[i]; os_ = np.sort(o[np.isfinite(o)])
        out.loc[i, "APR_DEP_20"] = np.where(np.isfinite(o), say(os_, o - 600, o + 600) - 1, np.nan)
        a = arr_g.get(k, np.array([]))
        out.loc[i, "APR_ARR_20"] = np.where(np.isfinite(o), say(a, o - 600, o + 600), np.nan)
    return out

def blok(df):
    df = df.copy(); df["T"] = K.ep(df["MVT_TIME_UTC_mvt"]); df["O"] = K.ep(df["AOBT_3_flt"])
    df["g"] = np.clip(df["T"] - K.ep(df["EOBT_1_flt"]), -3600, 14400)
    df["pg"] = grup(df["STAND_mvt"]); df["al"] = df["FLIGHT_mvt"].fillna("").astype(str).str.extract(r"^([A-Z0-9]{2}?[A-Z]?)", expand=False).fillna("")
    df["P"] = df["T"] + 480.0
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")]
    return pd.concat([apt_blok(d.reset_index(drop=True), arr[arr["ADES_mvt"].eq(a)]) for a, d in dep.groupby("ADEP_mvt")], ignore_index=True)

t0 = time.time()
tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)], ignore_index=True)
rk = pd.read_parquet(K.config.RANK_RAW, columns=KOL)
print("park grubu örnek:", grup(rk["STAND_mvt"].head(8)).tolist(), rk["STAND_mvt"].head(8).tolist())
out = pd.concat([blok(tr), blok(rk)], ignore_index=True)
assert out["MVT_ID_mvt"].is_unique
out.astype({c: "float32" for c in out.columns[1:]}).to_parquet(CIK, index=False)
print(f"{len(out):,} | {time.time() - t0:.0f} sn"); print(out.describe().T.round(2).to_string())

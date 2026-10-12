"""Gerçekleşen yüzey özellikleri (02.10 akşam). İzsiz havalimanları (LTFM, LFPG,
Temmuz LIRF) için ADS-B'nin yerini tutacak, testte açık sinyaller. Etiket (kalkışın BLOCK'u) kullanılmaz.

Kalkış anı T (MVT), aynı havalimanı:
  ARR_TI_W30_MED / _MAX / _N : [T − 30 dk, T] içinde inen uçakların taksi-in süresi (BLOCK − MVT; testte varış BLOCK'u
                               %100 açık). Mevcut ARR_TAXI_MED_30 NM off-block'tan ÖNCEKİ 30 dk'ya bakıyor; bu, kalkışın
                               kendi taksi penceresine bakıyor.
  ARR_TI_W15_MED, ARR_TI_F15_MED : [T − 15 dk, T] ve (T, T + 15 dk]
  NMT_C15_G, NMT_F15_G, NMT_PIST_C15_G : pencere_ileri_ozellik.py'deki NM taksisi ortancaları, yapay yığınlar dışarıda
                               (LTFM'de NM off-block'u satırların ~%94'ünde sabit taksi değerinden türetilmiş; yığın: bkz. yiginlar(); eski tanım 10 sn'lik
                               NM taksi kutularında payı > %3 olan kutu, ±15 sn).
  NMT_YAPAY : satırın kendisi yığında mı (0/1)
Çıktı: data/yuzey_oz.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K  # noqa: E402
from pencere_ileri_ozellik import med_pencere  # noqa: E402
from src import config  # noqa: E402

KOL = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "RUNWAY_mvt",
       "AOBT_3_flt"]


def yiginlar(nt):
    """Tam dakikalarda (±15 sn) payı > %3 ve komşu ±5 dakikanın ortancasının 3 katından fazla olan NM taksi değerleri.
    02.10: LTFM 960/1140/1200/1440/1500 (satırların ~%94'ü), EDDM 300/660–900, LEBL 600–720/1200/1260."""
    x = nt[np.isfinite(nt) & (nt > 0) & (nt < 7200)]
    if len(x) < 1000:
        return np.array([])
    c = np.arange(60, 3600, 60)
    sh = np.array([np.mean(np.abs(x - v) <= 15) for v in c])
    kom = np.array([np.median(np.r_[sh[max(0, k - 5):k], sh[k + 1:k + 6]]) for k in range(len(c))])
    return c[(sh / np.maximum(kom, 1e-4) > 3) & (sh > 0.03)].astype(float)


def ara_stat(At, Av, T, a_ofs, b_ofs, fn):
    """Sıralı At (varış anları) için her T: At ∈ [T + a_ofs, T + b_ofs] olan Av'lerin fn'i."""
    lo, hi = np.searchsorted(At, T + a_ofs, "left"), np.searchsorted(At, T + b_ofs, "right")
    out = np.full(len(T), np.nan)
    for i in range(len(T)):
        if hi[i] > lo[i]:
            out[i] = fn(Av[lo[i]:hi[i]])
    return out, (hi - lo).astype(float)


def blok(df, yig):
    df = df.copy()
    df["T"] = K.ep(df["MVT_TIME_UTC_mvt"])
    df["pist"] = df["RUNWAY_mvt"].astype(str)
    df["nt"] = np.clip(df["T"] - K.ep(df["AOBT_3_flt"]), 0, 7200)
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")].copy()
    arr["ti"] = np.clip(K.ep(arr["BLOCK_TIME_UTC_mvt"]) - arr["T"], 0, 7200)
    parca = []
    for a, d in dep.groupby("ADEP_mvt"):
        d = d.sort_values("T", kind="stable").reset_index(drop=True)
        T, nt, R = d["T"].to_numpy(float), d["nt"].to_numpy(float), d["pist"].to_numpy()
        merk = yig.get(a, np.array([]))
        yap = np.zeros(len(d), bool)
        for c in merk:
            yap |= np.abs(nt - c) <= 15
        ntg = np.where(yap, np.nan, nt)
        o = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()})
        o["NMT_YAPAY"] = yap.astype(float)
        o["NMT_C15_G"] = med_pencere(T, ntg, -900, 900)
        o["NMT_F15_G"] = med_pencere(T, ntg, 0, 900)
        o["NMT_PIST_C15_G"] = np.nan
        for r in np.unique(R):
            i = np.where(R == r)[0]
            o.loc[i, "NMT_PIST_C15_G"] = med_pencere(T[i], ntg[i], -900, 900)
        aa = arr[arr["ADES_mvt"].eq(a)].sort_values("T", kind="stable")
        At, Av = aa["T"].to_numpy(float), aa["ti"].to_numpy(float)
        ok = np.isfinite(Av)
        At, Av = At[ok], Av[ok]
        o["ARR_TI_W30_MED"], o["ARR_TI_W30_N"] = ara_stat(At, Av, T, -1800, 0, np.median)
        o["ARR_TI_W30_MAX"], _ = ara_stat(At, Av, T, -1800, 0, np.max)
        o["ARR_TI_W15_MED"], _ = ara_stat(At, Av, T, -900, 0, np.median)
        o["ARR_TI_F15_MED"], _ = ara_stat(At, Av, T, 1, 900, np.median)
        o["ARR_TI_W30_N"] = o["ARR_TI_W30_N"].fillna(0)
        parca.append(o)
    return pd.concat(parca, ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                             "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=KOL)
    # yığınlar eğitimden belirlenir (testte de aynı merkezler)
    dtr = tr[tr["PHASE_mvt"].eq("DEP")]
    nt = np.clip(K.ep(dtr["MVT_TIME_UTC_mvt"]) - K.ep(dtr["AOBT_3_flt"]), 0, 7200)
    yig = {a: yiginlar(nt[i]) for a, i in dtr.groupby("ADEP_mvt").indices.items()}
    print("yığınlar:", {a: v.tolist() for a, v in yig.items() if len(v)})
    out = pd.concat([blok(tr, yig), blok(rk, yig)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "yuzey_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T[["count", "mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()

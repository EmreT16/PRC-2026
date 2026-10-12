"""Varış yoğunluğu + METAR özellikleri (01.10 gece), `_kqgp` sürümünde pist bazlı gecikme endeksiyle birlikte.
Tarama (lgbh3_kqg OOF kalıntısı, 6 ay ay dışı; kontrol = GEC_* + GEC_PIST_*; gürültü tabanı ayda ±0,1):
  varış yoğunluğu −0,14 (6/6 ay, LTFM −0,76); METAR −0,23 (5/6; kış aylarında, Temmuz nötr); ikisi birlikte −0,34 (6/6).
Özellikler (kalkış anı T; havalimanı a, pist r):
  AR_PIST_15 / AR_APT_15 : [T − 15 dk, T) aralığında aynı piste / havalimanına inen uçak sayısı
  AR_PIST_PAY_60         : son 60 dk'daki varışlarda bu pistin payı (karışık kullanım vekili)
  WX_FZ_0, WX_FG_0, WX_SN_0, WX_SN_3S, WX_TS_0, WX_GORUS, WX_RUZGAR, WX_HAMLE : data/wx_oz.parquet'ten (kalkış saati;
  IEM METAR, kamu malı; scripts/wx_ozellik.py)
Etiket kullanılmaz; testte de hesaplanır (ranking.parquet).
Çıktı: data/varis_wx_oz.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K  # noqa: E402
from src import config  # noqa: E402

KOL = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]
WX = ["WX_FZ_0", "WX_FG_0", "WX_SN_0", "WX_SN_3S", "WX_TS_0", "WX_GORUS", "WX_RUZGAR", "WX_HAMLE"]


def say(s, a, b):
    """sıralı s'de a ≤ x < b olan eleman sayısı."""
    return np.searchsorted(s, b, "left") - np.searchsorted(s, a, "left")


def havalimani(dep, arr):
    T, R = dep["T"].to_numpy(float), dep["pist"].to_numpy()
    at = np.sort(arr["T"].to_numpy(float))
    out = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy(), "AR_APT_15": say(at, T - 900, T).astype(float),
                        "AR_PIST_15": np.nan, "AR_PIST_PAY_60": np.nan})
    a60 = say(at, T - 3600, T)
    for r in np.unique(R):
        i = np.where(R == r)[0]
        ar = np.sort(arr.loc[arr["pist"].eq(r), "T"].to_numpy(float))
        out.loc[i, "AR_PIST_15"] = say(ar, T[i] - 900, T[i])
        out.loc[i, "AR_PIST_PAY_60"] = np.where(a60[i] > 0, say(ar, T[i] - 3600, T[i]) / np.maximum(a60[i], 1), np.nan)
    return out


def blok(df):
    df = df.copy()
    df["T"] = K.ep(df["MVT_TIME_UTC_mvt"])
    df["pist"] = df["RUNWAY_mvt"].astype(str)
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")]
    return pd.concat([havalimani(d.reset_index(drop=True), arr[arr["ADES_mvt"].eq(a)]) for a, d in dep.groupby("ADEP_mvt")],
                     ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                             "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=KOL)
    out = pd.concat([blok(tr), blok(rk)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    out = out.merge(pd.read_parquet(config.DATA_DIR / "wx_oz.parquet", columns=["MVT_ID_mvt"] + WX), on="MVT_ID_mvt",
                    how="left", validate="one_to_one")
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "varis_wx_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T[["count", "mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()

"""Komşu kalkış aralıkları (02.10 gece). Aynı pistte ve havalimanında, kalkış anı T'ye göre
k'ıncı önceki / sonraki kalkışa süre (sn). Mevcut KQ_ONCEKI_PIST / KQ_SONRAKI_PIST yalnız k = 1.
  KOM_PP_k / KOM_PS_k (pist, k = 2, 3, 5, 10), KOM_AP_k / KOM_AS_k (havalimanı, k = 5, 10)
Yardımcı (özellik değil): KOM_T (kalkış anı, epoch), KOM_PIST — tarama_olay.py komşu tahmin özellikleri için.
Etiket kullanılmaz. Çıktı: data/komsu_oz.parquet
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

KOL = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]


def araliklar(T, ks):
    n, out = len(T), {}
    for k in ks:
        p, s = np.full(n, np.nan), np.full(n, np.nan)
        if n > k:
            p[k:] = T[k:] - T[:-k]
            s[:-k] = T[k:] - T[:-k]
        out[k] = (p, s)
    return out


def blok(df):
    df = df[df["PHASE_mvt"].eq("DEP")].copy()
    df["T"] = K.ep(df["MVT_TIME_UTC_mvt"])
    df["pist"] = df["RUNWAY_mvt"].astype(str)
    parca = []
    for a, d in df.groupby("ADEP_mvt"):
        d = d.sort_values("T", kind="stable").reset_index(drop=True)
        o = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "KOM_T": d["T"].to_numpy(), "KOM_PIST": d["pist"].to_numpy()})
        for k, (p, s) in araliklar(d["T"].to_numpy(float), (5, 10)).items():
            o[f"KOM_AP_{k}"], o[f"KOM_AS_{k}"] = p, s
        for k in (2, 3, 5, 10):
            o[f"KOM_PP_{k}"], o[f"KOM_PS_{k}"] = np.nan, np.nan
        for r, i in d.groupby("pist").indices.items():
            for k, (p, s) in araliklar(d["T"].to_numpy(float)[i], (2, 3, 5, 10)).items():
                o.loc[i, f"KOM_PP_{k}"], o.loc[i, f"KOM_PS_{k}"] = p, s
        parca.append(o)
    return pd.concat(parca, ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    fs = [K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                   "training_2025-12-01_2026-01-01.parquet") for m in range(1, 13)]
    out = pd.concat([blok(pd.concat([pd.read_parquet(f, columns=KOL) for f in fs], ignore_index=True)),
                     blok(pd.read_parquet(config.RANK_RAW, columns=KOL))], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    for c in out.columns:
        if c.startswith("KOM_") and c not in ("KOM_PIST",):
            out[c] = out[c].astype("float64" if c == "KOM_T" else "float32")
    out.to_parquet(config.DATA_DIR / "komsu_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()

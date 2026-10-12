"""Kayan gecikme endeksi (30.09): her kalkış için, kendisi hariç,
aynı havalimanında son 15 / 30 dk'da kalkan uçakların ortanca gecikmesi (kalkış − EOBT_1) ve son 60 dk'da tarifeye göre
ortanca gecikmesi (kalkış − SCHED). Model A uçuşun kendi gecikmesini görüyor, çevresindeki gecikme rejimini görmüyor.
Etiket kullanılmaz; testte de hesaplanır (ranking.parquet).
Çıktı: data/gecikme_oz.parquet (MVT_ID_mvt, GEC_EOBT_G15, GEC_EOBT_G30, GEC_SCHED_G60, GEC_N_G15, GEC_KENDI_FARK).
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

KOL = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt"]
ALT, UST = -3600.0, 14400.0


def pencere_medyan(T, v, W):
    """Her i için T_j ∈ [T_i − W, T_i) (aynı andaki kalkışlar ve kendisi hariç) v_j'lerin ortancası ve sayısı."""
    lo = np.searchsorted(T, T - W, "left")
    hi = np.searchsorted(T, T, "left")
    med, n = np.full(len(T), np.nan), np.zeros(len(T))
    for i in range(len(T)):
        a = v[lo[i]:hi[i]]
        a = a[np.isfinite(a)]
        if len(a):
            med[i], n[i] = np.median(a), len(a)
    return med, n


def blok(df):
    df = df[df["PHASE_mvt"].eq("DEP")].copy()
    df["T"] = K.ep(df["MVT_TIME_UTC_mvt"])
    df["g_eobt"] = np.clip(df["T"] - K.ep(df["EOBT_1_flt"]), ALT, UST)
    df["g_sched"] = np.clip(df["T"] - K.ep(df["SCHED_TIME_UTC_mvt"]), ALT, UST)
    parca = []
    for _, d in df.groupby("ADEP_mvt"):
        d = d.sort_values("T", kind="stable")
        T = d["T"].to_numpy(float)
        e, s = d["g_eobt"].to_numpy(float), d["g_sched"].to_numpy(float)
        g15, n15 = pencere_medyan(T, e, 900.0)
        g30, _ = pencere_medyan(T, e, 1800.0)
        s60, _ = pencere_medyan(T, s, 3600.0)
        parca.append(pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "GEC_EOBT_G15": g15, "GEC_EOBT_G30": g30,
                                   "GEC_SCHED_G60": s60, "GEC_N_G15": n15, "GEC_KENDI_FARK": e - g30}))
    return pd.concat(parca, ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                             "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=KOL)
    out = pd.concat([blok(tr), blok(rk)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "gecikme_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T.round(1).to_string())


if __name__ == "__main__":
    main()

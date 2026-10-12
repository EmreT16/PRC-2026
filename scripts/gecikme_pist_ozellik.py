"""Pist bazlı kayan gecikme endeksi (01.10 gece).
gecikme_ozellik.py'nin havalimanı geneli endeksine ek: aynı havalimanı + aynı pist (RUNWAY_mvt; boş pist kendi grubu),
son 15 / 30 dk, kendisi ve aynı andakiler hariç, ortanca (kalkış − EOBT_1), [−3.600, 14.400] sn'ye kırpık.
Ön araştırmadaki ay dışı doğrusal sınama: −0,30 sn, 60 havalimanı-ayının 48'i iyi, LTFM 6/6 (belirsiz: ağaç modelde
kazanç ayrıca ölçülmeli). Etiket kullanılmaz; testte de hesaplanır (ranking.parquet).
Çıktı: data/gecikme_pist_oz.parquet (MVT_ID_mvt, GEC_PIST_EOBT_G15, GEC_PIST_EOBT_G30, GEC_PIST_N_G15, GEC_PIST_FARK_G15).
GEC_PIST_FARK_G15 = GEC_PIST_EOBT_G15 − GEC_EOBT_G15 (data/gecikme_oz.parquet; pist ile havalimanı farkı).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K  # noqa: E402
from gecikme_ozellik import ALT, UST, pencere_medyan  # noqa: E402
from src import config  # noqa: E402

KOL = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "EOBT_1_flt"]


def blok(df):
    df = df[df["PHASE_mvt"].eq("DEP")].copy()
    df["T"] = K.ep(df["MVT_TIME_UTC_mvt"])
    df["g_eobt"] = np.clip(df["T"] - K.ep(df["EOBT_1_flt"]), ALT, UST)
    df["pist"] = df["RUNWAY_mvt"].astype(str)
    parca = []
    for _, d in df.groupby(["ADEP_mvt", "pist"]):
        d = d.sort_values("T", kind="stable")
        T, e = d["T"].to_numpy(float), d["g_eobt"].to_numpy(float)
        g15, n15 = pencere_medyan(T, e, 900.0)
        g30, _ = pencere_medyan(T, e, 1800.0)
        parca.append(pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "GEC_PIST_EOBT_G15": g15,
                                   "GEC_PIST_EOBT_G30": g30, "GEC_PIST_N_G15": n15}))
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
    g = pd.read_parquet(config.DATA_DIR / "gecikme_oz.parquet", columns=["MVT_ID_mvt", "GEC_EOBT_G15"])
    out = out.merge(g, on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert out["GEC_EOBT_G15"].notna().sum() >= out["GEC_PIST_EOBT_G15"].notna().sum()
    out["GEC_PIST_FARK_G15"] = out["GEC_PIST_EOBT_G15"] - out["GEC_EOBT_G15"]
    out = out.drop(columns="GEC_EOBT_G15")
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "gecikme_pist_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T.round(1).to_string())


if __name__ == "__main__":
    main()

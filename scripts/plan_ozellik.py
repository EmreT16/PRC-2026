"""Planlı saat farkları (30.09): LOBT / IOBT / EOBT_1 − AOBT_3 (sn) — E1'in planlı saat yankısı sınıfları için (aday e1p).
Çıktı: data/plan_oz.parquet (MVT_ID_mvt, PL_LOBT, PL_IOBT, PL_EOBT); 2025'in 12 ayı + 2026 testi, yalnız kalkışlar."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

RAW = config.PRC2_DIR / "data"
KOL = ["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt"]


def main():
    fs = [RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else "training_2025-12-01_2026-01-01.parquet")
          for m in range(1, 13)] + [config.RANK_RAW]
    d = pd.concat([pd.read_parquet(f, columns=KOL) for f in fs], ignore_index=True)
    d = d[d["PHASE_mvt"].eq("DEP")]
    out = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"]})
    for c, ad in (("LOBT_flt", "PL_LOBT"), ("IOBT_flt", "PL_IOBT"), ("EOBT_1_flt", "PL_EOBT")):
        out[ad] = (d[c] - d["AOBT_3_flt"]).dt.total_seconds().astype("float32")
    assert out["MVT_ID_mvt"].is_unique
    out.to_parquet(config.DATA_DIR / "plan_oz.parquet", index=False)
    print(len(out), out.describe().round(0).to_string())


if __name__ == "__main__":
    main()

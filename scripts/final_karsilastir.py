"""12.10: iki final dosyasını karşılaştırır — teslim edilen dosya ↔ sıfırdan yeniden üretilen
dosya (scripts/final_tum.sh başka bir klasörde). Satır kimlikleri aynı sırada olmalı; fark tür (A = NM'li, B = NM'siz),
ay ve havalimanı başına RMS / ortalama / en büyük ve |fark| > 60 sn payı olarak yazılır.
Kullanım: PRC_FINAL=1 python scripts/final_karsilastir.py <teslim.parquet> <yeniden.parquet>
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


def ozet(g):
    return pd.Series({"n": len(g), "rms": np.sqrt((g.d ** 2).mean()), "ort": g.d.mean(), "maks": g.d.abs().max(),
                      "pay_60sn_%": 100 * (g.d.abs() > 60).mean()})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    assert config.FINAL, "PRC_FINAL=1 gerekli"
    f1, f2 = sys.argv[1], sys.argv[2]
    for f in (f1, f2):
        print(f, "MD5", hashlib.md5(Path(f).read_bytes()).hexdigest(), "sha256", hashlib.sha256(Path(f).read_bytes()).hexdigest())
    a, t = pd.read_parquet(f1), pd.read_parquet(f2)
    assert len(a) == len(t) == config.TEST_N and np.array_equal(a.MVT_ID_mvt.to_numpy(), t.MVT_ID_mvt.to_numpy())
    d = t.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    r = a[["MVT_ID_mvt"]].merge(pd.read_parquet(config.RANK_RAW, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]),
                                on="MVT_ID_mvt", how="left", validate="one_to_one")
    x = pd.DataFrame({"havalimani": r.ADEP_mvt, "ay": r.MVT_TIME_UTC_mvt.dt.month,
                      "tur": np.where(r.AOBT_3_flt.isna(), "B", "A"), "d": d})
    print(f"tüm: n {len(d):,} | RMS {np.sqrt(np.mean(d ** 2)):.2f} | ort {d.mean():+.3f} | maks {np.abs(d).max():.0f} | "
          f"|fark| > 60 sn %{100 * np.mean(np.abs(d) > 60):.2f} | korelasyon "
          f"{np.corrcoef(a.TAXITIME_SEC_mvt, t.TAXITIME_SEC_mvt)[0, 1]:.6f}")
    for k in ("tur", "ay", "havalimani"):
        print(x.groupby(k)[["d"]].apply(ozet).round(2).to_string())


if __name__ == "__main__":
    main()

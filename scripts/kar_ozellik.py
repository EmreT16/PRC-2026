"""Kar / buzlanma özellikleri (02.10 akşam). Kaynak: IEM METAR saatlik
(extra_data/metar_saatlik.parquet, kamu malı). Ocak 2026, 2025 kışından çok daha karlı (EDDF 11 → 70 kar saati).
Kalkışın MVT saati (UTC, saat başına yuvarlanmış) ile aynı havalimanının METAR'ı:
  KAR_3S/6S/12S/24S : son 3/6/12/24 saatte kar saati (o saat dahil); DON_6S : son 6 saatte donan yağış saati
  KAR_SONRA_3S : sonraki 3 saatte kar saati; KAR_GUN / KAR_ONCEKI_GUN : o UTC günü / bir önceki gün kar saati
  SIC_0, SIC_MIN12, SPREAD_0 : sıcaklık, 12 sa en düşük sıcaklık, sıcaklık − çiy noktası
Çıktı: data/kar_oz.parquet
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

METAR = config.EK_VERI / "metar_saatlik.parquet"
KOL = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]


def saatlik():
    m = pd.read_parquet(METAR)
    parca = []
    for icao, x in m.groupby("ICAO"):
        x = x.set_index("saat").sort_index()
        x = x[~x.index.duplicated()].reindex(pd.date_range(x.index.min(), x.index.max(), freq="h"))
        sn, fz = x["WX_SN"].astype(float).fillna(0), x["WX_FZ"].astype(float).fillna(0)
        o = pd.DataFrame(index=x.index)
        for k in (3, 6, 12, 24):
            o[f"KAR_{k}S"] = sn.rolling(k, min_periods=1).sum()
        o["DON_6S"] = fz.rolling(6, min_periods=1).sum()
        o["KAR_SONRA_3S"] = sn[::-1].rolling(3, min_periods=1).sum()[::-1].shift(-1).fillna(0)
        gun = sn.groupby(sn.index.normalize()).sum()
        o["KAR_GUN"] = gun.reindex(o.index.normalize()).to_numpy()
        o["KAR_ONCEKI_GUN"] = gun.shift(1, freq="D").reindex(o.index.normalize()).to_numpy()
        t = x["SICAKLIK_C"].astype(float).interpolate(limit=3)
        o["SIC_0"] = t
        o["SIC_MIN12"] = t.rolling(12, min_periods=1).min()
        o["SPREAD_0"] = t - x["CIY_NOKTASI_C"].astype(float).interpolate(limit=3)
        o["ADEP_mvt"], o["saat"] = icao, o.index
        parca.append(o.reset_index(drop=True))
    return pd.concat(parca, ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    fs = [K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                   "training_2025-12-01_2026-01-01.parquet") for m in range(1, 13)] + [config.RANK_RAW]
    d = pd.concat([pd.read_parquet(f, columns=KOL) for f in fs], ignore_index=True)
    d = d[d["PHASE_mvt"].eq("DEP")]
    d["saat"] = d["MVT_TIME_UTC_mvt"].dt.tz_convert(None).dt.floor("h")
    s = saatlik()
    out = d[["MVT_ID_mvt", "ADEP_mvt", "saat"]].merge(s, on=["ADEP_mvt", "saat"], how="left").drop(columns=["ADEP_mvt", "saat"])
    assert out["MVT_ID_mvt"].is_unique and len(out) == len(d)
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "kar_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn | eşleşme %{out['KAR_3S'].notna().mean() * 100:.1f}")
    print(out.describe().T[["count", "mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()

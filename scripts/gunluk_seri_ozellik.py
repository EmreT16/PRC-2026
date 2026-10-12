"""EUROCONTROL günlük serileri (04.10).
Kaynak: https://www.eurocontrol.int/performance/data/download/csv/{atfm_slot_adherence, all_pre_departure_delays,
atc_pre_departure_delays}_{2025,2026}.csv (indirilen kopyalar extra_data/ansperf/). (UTC gün, havalimanı):
  GS_REG_PAY   : slotlu (ATFM düzenlemeli) kalkış payı
  GS_GEC_PAY   : slotlu kalkışlarda slot penceresinden geç çıkanların payı
  GS_ERK_PAY   : ... erken çıkanların payı
  GS_ONCE_DK   : kalkış başına kalkış öncesi gecikme (tüm nedenler, dk)
  GS_ATC_DK    : kalkış başına ATC kalkış öncesi gecikmesi (dk)
Çıktı: data/gunluk_oz.parquet"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

KAY = config.EK_VERI / "ansperf"
HAM = config.ROOT.parent / "data"


def seri(ad, sut):
    d = pd.concat([pd.read_csv(KAY / f"{ad}_{y}.csv", usecols=["FLT_DATE", "APT_ICAO"] + sut) for y in (2025, 2026)])
    d["gun"] = pd.to_datetime(d.FLT_DATE).dt.strftime("%Y-%m-%d")
    return d.groupby(["gun", "APT_ICAO"])[sut].sum(min_count=1)


s1 = seri("atfm_slot_adherence", ["FLT_DEP_1", "FLT_DEP_REG_1", "FLT_DEP_OUT_EARLY_1", "FLT_DEP_OUT_LATE_1"])
s2 = seri("all_pre_departure_delays", ["FLT_DEP_IFR_2", "DLY_ALL_PRE_2"])
s3 = seri("atc_pre_departure_delays", ["FLT_DEP_3", "DLY_ATC_PRE_3"])
g = s1.join(s2, how="outer").join(s3, how="outer")
g["GS_REG_PAY"] = g.FLT_DEP_REG_1 / g.FLT_DEP_1
g["GS_GEC_PAY"] = g.FLT_DEP_OUT_LATE_1 / g.FLT_DEP_REG_1.where(g.FLT_DEP_REG_1 > 0)
g["GS_ERK_PAY"] = g.FLT_DEP_OUT_EARLY_1 / g.FLT_DEP_REG_1.where(g.FLT_DEP_REG_1 > 0)
g["GS_ONCE_DK"] = g.DLY_ALL_PRE_2 / g.FLT_DEP_IFR_2.where(g.FLT_DEP_IFR_2 > 0)
g["GS_ATC_DK"] = g.DLY_ATC_PRE_3 / g.FLT_DEP_3.where(g.FLT_DEP_3 > 0)
SUT = ["GS_REG_PAY", "GS_GEC_PAY", "GS_ERK_PAY", "GS_ONCE_DK", "GS_ATC_DK"]
g = g[SUT].reset_index()

c = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]
EGITIM = sorted(glob.glob(str(HAM / "training_2025-*.parquet")))
assert len(EGITIM) == 12, f"{HAM}: 12 eğitim dosyası beklenir, {len(EGITIM)} bulundu"
m = pd.concat([pd.read_parquet(f, columns=c) for f in EGITIM + [str(config.RANK_RAW)]])
m = m[m.PHASE_mvt == "DEP"]
m["gun"] = pd.to_datetime(m.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
o = m[["MVT_ID_mvt", "gun", "ADEP_mvt"]].merge(g, left_on=["gun", "ADEP_mvt"], right_on=["gun", "APT_ICAO"], how="left")
o["MVT_ID_mvt"] = o.MVT_ID_mvt.astype("float64")
assert o.MVT_ID_mvt.is_unique
o[["MVT_ID_mvt"] + SUT].astype({k: "float32" for k in SUT}).to_parquet(config.DATA_DIR / "gunluk_oz.parquet", index=False)
o["yil"] = o.gun.str[:4]
print(len(o), "| dolu %:", (o[SUT].notna().mean() * 100).round(1).to_dict())
print(o.groupby("yil")[SUT].mean().round(3).T.to_string())

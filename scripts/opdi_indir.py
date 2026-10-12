"""OPDI v0.0.2 uçuş olaylarını (EUROCONTROL, 10 günlük Parquet) indirir ve 10 havalimanının yer olaylarına süzer.

07.10: 04.10'daki kapsama ölçümü yalnız adsb.lol izi hiç olmayan satırlara bakmıştı.
EDDF 2026'da kalkışların %32'si adsb.lol'da yalnız kalkış koşusunda görülüyor (2025'te %8,5); 20–28 Tem 2026'da bu
satırların %74'ünde OPDI uçağı yerde (park çıkışı / taxiway / apron) görüyor, LSZH'de %77. Diğer havalimanlarında ~%0.
Kaynak OpenSky'dan türetilmiş olay verisi
(state vector / Trino kullanılmıyor).

Pencereler 10 günlük, 2026-01-10 ızgarasında (…, 20250105_20250115, …, 20260719_20260729, …).
Süzme: yalnız yer olayları + take-off + first_seen; havalimanı referansına (OPDI pist olaylarının medyanı) ≤ 8 km.
Çıktı: PRC_adsb/opdi/suz/fe_<baş>_<son>.parquet (flight_id, type, t (epoch sn), apt, osm_ref, kullanim_s, icao24, cs).
Ham dosya süzüldükten sonra silinir. Yeniden çalıştırınca var olan çıktıları atlar.
Kullanım: python scripts/opdi_indir.py [2024-12-26 2026-08-08]
"""
import datetime as dt
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(os.environ.get("PRC_ADSB_KOK", str(Path(__file__).resolve().parents[2] / "PRC_adsb"))) / "opdi"
SUZ = KOK / "suz"
URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{}_{}.parquet"
TIPLER = {"exit-parking_position", "entry-parking_position", "entry-taxiway", "exit-taxiway", "entry-apron", "exit-apron",
          "entry-runway", "exit-runway", "entry-threshold", "exit-threshold", "take-off", "first_seen",
          "entry-hangar", "exit-hangar"}
# OPDI take-off / entry-runway olaylarının medyanı (flight_events_20260719_20260729_10apt.parquet)
APT = {"EDDF": (50.0378, 8.5391), "EDDM": (48.3430, 11.7739), "EGLL": (51.4649, -0.4530), "EHAM": (52.3184, 4.7385),
       "LEBL": (41.2919, 2.0823), "LEMD": (40.4990, -3.5592), "LFPG": (49.0215, 2.5550), "LIRF": (41.8020, 12.2412),
       "LSZH": (47.4575, 8.5588), "LTFM": (41.2837, 28.7272)}
YARICAP_M = 8_000.0
TEST_AYLAR_2026 = (1, 2, 6, 7)


def pencereler(bas, son):
    k = dt.date(2026, 1, 10)
    while k > bas:
        k -= dt.timedelta(days=10)
    while k < son:
        yield k, k + dt.timedelta(days=10)
        k += dt.timedelta(days=10)


def suz(ham):
    d = pd.read_parquet(ham, columns=["flight_id", "type", "event_time", "longitude", "latitude", "info"])
    d = d[d["type"].isin(TIPLER)].reset_index(drop=True)
    lat, lon = d["latitude"].to_numpy(float), d["longitude"].to_numpy(float)
    en, apt = np.full(len(d), np.inf), np.full(len(d), None, dtype=object)
    for a, (la, lo) in APT.items():
        m = 111_195.0 * np.hypot(lat - la, (lon - lo) * np.cos(np.radians(la)))
        y = m < en
        en[y], apt[y] = m[y], a
    d["apt"] = apt
    d = d[en <= YARICAP_M].reset_index(drop=True)
    d["t"] = (pd.to_datetime(d["event_time"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    inf = d["info"].fillna("")
    d["osm_ref"] = inf.str.extract(r'"osm_ref":"([^"]*)"')[0]
    d["kullanim_s"] = pd.to_numeric(inf.str.extract(r'"opdi_time_in_use_s":(-?[0-9.]+)')[0], errors="coerce")
    d["icao24"] = inf.str.extract(r'"osn_icao24":"([^"]*)"')[0]
    d["cs"] = inf.str.extract(r'"osn_flight_id":"([^"]*)"')[0].str.strip()
    return d[["flight_id", "type", "t", "apt", "latitude", "longitude", "osm_ref", "kullanim_s", "icao24", "cs"]]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    bas = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2024, 12, 26)
    son = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2026, 8, 8)
    SUZ.mkdir(parents=True, exist_ok=True)
    for a, b in pencereler(bas, son):
        ad = f"{a:%Y%m%d}_{b:%Y%m%d}"
        cik = SUZ / f"fe_{ad}.parquet"
        if cik.exists():
            continue
        # 2025 Şubat–Mayıs, Eylül–Kasım da eğitim (12 aylık düzeltici); 2026'da yalnız test ayları gerekli
        # (08.10: final aşaması Ocak + Şubat + Haziran + Temmuz 2026)
        if a.year == 2026 and not any(b >= dt.date(2026, m, 1) and a < (dt.date(2026, m + 1, 1)) for m in TEST_AYLAR_2026):
            continue
        ham = KOK / f"ham_{ad}.parquet"
        t0 = time.time()
        for deneme in range(4):
            try:
                urllib.request.urlretrieve(URL.format(f"{a:%Y%m%d}", f"{b:%Y%m%d}"), ham)
                break
            except Exception as e:  # noqa: BLE001
                print(f"{ad}: indirme hatası ({e}); deneme {deneme + 1}", flush=True)
                time.sleep(20)
        else:
            print(f"{ad}: İNDİRİLEMEDİ", flush=True)
            continue
        mb = ham.stat().st_size / 1e6
        d = suz(ham)
        d.to_parquet(cik, index=False)
        ham.unlink()
        print(f"{ad}: {mb:,.0f} MB → {len(d):,} olay ({d['apt'].value_counts().to_dict()}) | {time.time() - t0:.0f} sn",
              flush=True)


if __name__ == "__main__":
    main()

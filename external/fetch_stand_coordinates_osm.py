"""
OpenStreetMap Overpass API — Stand Koordinatları Çekme
10 havalimanı için aeroway=parking_position tagını çeker
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

BASE_PRC4 = os.environ["PRC4_DIR"]
OUT_STANDS = os.path.join(BASE_PRC4, "data", "features", "stand_coordinates.csv")
OUT_DIST   = os.path.join(BASE_PRC4, "data", "features", "stand_rwy_distances.csv")

AIRPORT_BBOX = {
    "EDDF": (49.95, 50.07, 8.47, 8.64),
    "EDDM": (48.31, 48.40, 11.74, 11.86),
    "EGLL": (51.45, 51.49, -0.51, -0.42),
    "EHAM": (52.28, 52.34, 4.70, 4.82),
    "LEBL": (41.27, 41.32, 2.03, 2.13),
    "LEMD": (40.45, 40.55, -3.63, -3.52),
    "LFPG": (48.98, 49.04, 2.51, 2.60),
    "LIRF": (41.77, 41.83, 12.21, 12.30),
    "LSZH": (47.44, 47.48, 8.53, 8.58),
    "LTFM": (41.25, 41.31, 28.70, 28.78),
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "PRC2026-StandCoord/1.0 (aviation research)",
    "Accept": "application/json",
})

all_stands = []

for apt, (lat_min, lat_max, lon_min, lon_max) in AIRPORT_BBOX.items():
    print(f"[{apt}] Stand koordinatlari cekiliyor...")
    query = f"""[out:json][timeout:30];
(
  node["aeroway"="parking_position"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["aeroway"="parking_position"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out center;"""

    try:
        resp = SESSION.get(OVERPASS_URL, params={"data": query}, timeout=40)
        print(f"  HTTP {resp.status_code}")
        if resp.status_code != 200:
            # Try POST
            resp = SESSION.post(OVERPASS_URL, data={"data": query}, timeout=40)
            print(f"  POST HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(f"  ATLIYOR: {apt}")
            time.sleep(2)
            continue

        data = resp.json()
        elements = data.get("elements", [])
        apt_count = 0
        for el in elements:
            ref = el.get("tags", {}).get("ref", el.get("tags", {}).get("name", ""))
            el_type = el.get("type", "")
            if el_type == "node":
                lat, lon = el.get("lat"), el.get("lon")
            elif el_type == "way":
                center = el.get("center", {})
                lat, lon = center.get("lat"), center.get("lon")
            else:
                continue
            if lat and lon and ref:
                all_stands.append({"ADEP": apt, "STAND_ID": str(ref).strip(),
                                   "LAT": float(lat), "LON": float(lon)})
                apt_count += 1
        print(f"  -> {apt_count} stand")
        time.sleep(2)
    except Exception as e:
        print(f"  HATA: {e}")
        time.sleep(3)

df_stands = pd.DataFrame(all_stands)
print(f"\nToplam stand: {len(df_stands)}")
if len(df_stands) > 0:
    print(df_stands.groupby("ADEP")["STAND_ID"].count().to_string())
    os.makedirs(os.path.dirname(OUT_STANDS), exist_ok=True)
    df_stands.to_csv(OUT_STANDS, index=False)
    print(f"Kaydedildi: {OUT_STANDS}")

    # Pist koordinatları
    df_rwy = pd.read_csv(os.path.join(BASE_PRC4, "data", "features", "external_runways.csv"))
    airports_10 = list(AIRPORT_BBOX.keys())
    sub_rwy = df_rwy[df_rwy['airport_ident'].isin(airports_10)].copy()
    rwy_coords = {}
    for _, r in sub_rwy.iterrows():
        apt2 = r['airport_ident']
        for side in ['le', 'he']:
            ident = str(r.get(f'{side}_ident', '')).strip()
            lat2  = r.get(f'{side}_latitude_deg')
            lon2  = r.get(f'{side}_longitude_deg')
            if ident and ident != 'nan' and pd.notna(lat2) and pd.notna(lon2):
                rwy_coords[(apt2, ident)] = (float(lat2), float(lon2))

    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000
        p1, p2 = radians(lat1), radians(lat2)
        dp = radians(lat2 - lat1); dl = radians(lon2 - lon1)
        a = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
        return 2*R*asin(sqrt(a))

    dist_records = []
    for apt2 in airports_10:
        apt_stands = df_stands[df_stands["ADEP"] == apt2]
        apt_rwys = {k[1]: v for k, v in rwy_coords.items() if k[0] == apt2}
        if len(apt_stands) == 0 or len(apt_rwys) == 0:
            continue
        for _, sr in apt_stands.iterrows():
            for rwy_id, (rlat, rlon) in apt_rwys.items():
                dist_m = haversine_m(sr["LAT"], sr["LON"], rlat, rlon)
                dist_records.append({"ADEP": apt2, "STAND_ID": sr["STAND_ID"],
                                     "RUNWAY": rwy_id, "DIST_M": dist_m,
                                     "DIST_S": dist_m / 5.0})

    df_dist = pd.DataFrame(dist_records)
    df_dist.to_csv(OUT_DIST, index=False)
    print(f"Mesafe tablosu: {OUT_DIST} ({len(df_dist):,} satır)")

    # LIRF özet
    print("\nLIRF stand→runway (ilk 5 stand, tüm pistler):")
    lirf_dist = df_dist[df_dist["ADEP"]=="LIRF"]
    for stand in lirf_dist["STAND_ID"].unique()[:5]:
        print(f"  Stand {stand}:")
        for _, r2 in lirf_dist[lirf_dist["STAND_ID"]==stand].iterrows():
            print(f"    -> RWY {r2['RUNWAY']:5s}: {r2['DIST_M']:6.0f}m (~{r2['DIST_S']:.0f}s)")
else:
    print("Hic stand bulunamadi! Manuel kontrol gerekli.")
    # Alternatif: coordinates.csv yoksa simule et
    print("\nAlternatif: Pist merkez koordinatlarını havalimanı anchor olarak kullan")

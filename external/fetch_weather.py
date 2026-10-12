"""
Havalimani bazli saatlik hava durumu verisi (Open-Meteo Archive API).

Onceki surumden farklari:
  - Ek degiskenler: cig noktasi (de-icing), ruzgar YONU (pist ile aci farki icin),
    kar derinligi, bulut orani, bagil nem.
  - Merge anahtarina YEAR eklendi (onceki surum (ay,gun,saat) kullaniyordu; tek yil
    oldugu icin calisiyordu ama 2025+2026 birlikte islenince cakisirdi).
  - Hem egitim (2025) hem siralama (2026) donemi tek dosyaya yaziliyor.
"""
import pandas as pd
import numpy as np
import requests
import time

AIRPORTS = {
    'EDDF': (50.0333, 8.5705),
    'EDDM': (48.3537, 11.7860),
    'EGLL': (51.4700, -0.4543),
    'EHAM': (52.3105, 4.7683),
    'LEBL': (41.2974, 2.0833),
    'LEMD': (40.4839, -3.5680),
    'LFPG': (49.0097, 2.5479),
    'LIRF': (41.7999, 12.2462),
    'LTFM': (41.2590, 28.7424),
    'LSZH': (47.4647, 8.5492),
}

HOURLY_VARS = [
    'temperature_2m', 'dew_point_2m', 'relative_humidity_2m',
    'precipitation', 'snowfall', 'snow_depth',
    'wind_speed_10m', 'wind_direction_10m', 'cloud_cover',
]
# NOT: 'visibility' bilincli olarak ISTENMIYOR. Archive (ERA5) uc noktasi bu degiskeni
# dondurmuyor - istek hata vermiyor, sutun sessizce bos geliyor. Gorus verisi
# METAR arsivinden ayrica cekiliyor.

RENAME = {
    'temperature_2m': 'TEMPERATURE',
    'dew_point_2m': 'DEW_POINT',
    'relative_humidity_2m': 'HUMIDITY',
    'precipitation': 'PRECIPITATION',
    'snowfall': 'SNOWFALL',
    'snow_depth': 'SNOW_DEPTH',
    'wind_speed_10m': 'WIND_SPEED',
    'wind_direction_10m': 'WIND_DIR',
    'cloud_cover': 'CLOUD_COVER',
}


def fetch_airport(icao, lat, lon, start_date, end_date):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(HOURLY_VARS)}"
    )
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200 and 'hourly' in resp.json():
                df = pd.DataFrame(resp.json()['hourly'])
                df['CURRENT_AIRPORT'] = icao
                df['time'] = pd.to_datetime(df['time'])
                return df
            print(f"  {icao}: status {resp.status_code}, tekrar deneniyor...")
        except Exception as e:
            print(f"  {icao}: {e}, tekrar deneniyor...")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{icao} icin hava durumu cekilemedi")


def add_derived(df):
    """Ham saatlik olculerden operasyonel sinyaller uretir."""
    df = df.sort_values(['CURRENT_AIRPORT', 'time']).copy()

    # De-icing riski: cig noktasi ile sicaklik yakin VE sicaklik dusukse ucakta buzlanma olur.
    df['FROST_RISK'] = ((df['TEMPERATURE'] <= 3) &
                        ((df['TEMPERATURE'] - df['DEW_POINT']).abs() <= 3)).astype(int)
    df['IS_FREEZING'] = (df['TEMPERATURE'] <= 0).astype(int)

    # Anlik kar yagisi degil, PISTTE BIRIKEN kar onemli -> son 3 ve 6 saatin toplami.
    g = df.groupby('CURRENT_AIRPORT', sort=False)
    df['SNOWFALL_3H'] = g['SNOWFALL'].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df['SNOWFALL_6H'] = g['SNOWFALL'].transform(lambda s: s.rolling(6, min_periods=1).sum())
    df['PRECIP_3H'] = g['PRECIPITATION'].transform(lambda s: s.rolling(3, min_periods=1).sum())
    
    # 3 saat oncesine gore sicaklik dususu (Pozitif deger = Hava X derece sogudu)
    df['TEMP_DROP_3H'] = g['TEMPERATURE'].transform(lambda s: s.shift(3) - s).fillna(0)

    # Ruzgar vektor bilesenleri: pist yonu ile aci farkini hesaplamak icin kullanilacak.
    rad = np.deg2rad(df['WIND_DIR'])
    df['WIND_U'] = -df['WIND_SPEED'] * np.sin(rad)
    df['WIND_V'] = -df['WIND_SPEED'] * np.cos(rad)

    df['MVT_YEAR'] = df['time'].dt.year
    df['MVT_MONTH'] = df['time'].dt.month
    df['MVT_DAY'] = df['time'].dt.day
    df['MVT_HOUR'] = df['time'].dt.hour
    return df.drop(columns=['time'])


if __name__ == "__main__":
    # Egitim donemi (2025 tamami) + siralama donemi (2026 Ocak ve Temmuz).
    periods = [("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-07-31")]

    frames = []
    for icao, (lat, lon) in AIRPORTS.items():
        for start, end in periods:
            print(f"{icao} {start} -> {end}")
            frames.append(fetch_airport(icao, lat, lon, start, end))
            time.sleep(1)

    weather = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    weather = add_derived(weather)
    weather = weather.drop_duplicates(subset=['CURRENT_AIRPORT', 'MVT_YEAR', 'MVT_MONTH',
                                              'MVT_DAY', 'MVT_HOUR'])

    out = "data/weather_all.parquet"
    weather.to_parquet(out, index=False)
    print(f"\nKAYDEDILDI: {out}  ({weather.shape[0]} satir, {weather.shape[1]} sutun)")
    print(weather.groupby('MVT_YEAR').size())

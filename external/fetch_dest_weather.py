import os
import pandas as pd
import requests
import time
import airportsdata

print("Open-Meteo üzerinden iniş havalimanları için indirme başlıyor...")

train_path = os.path.join(os.environ["PRC_DIR"], "data", "dep_train.parquet")
rank_path = os.path.join(os.environ["PRC_DIR"], "data", "dep_rank.parquet")

df_train = pd.read_parquet(train_path, columns=['ADES_mvt'])
df_rank = pd.read_parquet(rank_path, columns=['ADES_mvt'])
all_ades = set(df_train['ADES_mvt'].unique()).union(set(df_rank['ADES_mvt'].unique()))

AIRPORTS = airportsdata.load('icao')
start_date = "2025-01-01"
end_date = "2026-07-31"

HOURLY_VARS = [
    'temperature_2m', 'dew_point_2m', 'relative_humidity_2m', 
    'precipitation', 'snowfall', 'snow_depth', 'wind_speed_10m', 'visibility'
]

results = []
count = 0
for icao in all_ades:
    count += 1
    if icao not in AIRPORTS:
        continue
    lat, lon = AIRPORTS[icao]['lat'], AIRPORTS[icao]['lon']
    
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(HOURLY_VARS)}"
    )
    
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.DataFrame(resp.json()['hourly'])
                df['ADES_mvt'] = icao
                df['time'] = pd.to_datetime(df['time'])
                results.append(df)
                break
        except Exception:
            time.sleep(2)
            
    if count % 100 == 0:
        print(f"{count}/{len(all_ades)} tamamlandi...")

if len(results) > 0:
    final_df = pd.concat(results, ignore_index=True)
    final_df['MVT_YEAR'] = final_df['time'].dt.year
    final_df['MVT_MONTH'] = final_df['time'].dt.month
    final_df['MVT_DAY'] = final_df['time'].dt.day
    final_df['MVT_HOUR'] = final_df['time'].dt.hour
    final_df = final_df.drop(columns=['time'])
    
    final_df.rename(columns={
        'temperature_2m': 'DEST_TEMP',
        'dew_point_2m': 'DEST_DEW_POINT',
        'relative_humidity_2m': 'DEST_HUMIDITY',
        'precipitation': 'DEST_PRECIP',
        'snowfall': 'DEST_SNOW',
        'snow_depth': 'DEST_SNOW_DEPTH',
        'wind_speed_10m': 'DEST_WIND',
        'visibility': 'DEST_VISIBILITY'
    }, inplace=True)
    
    # Yeni Turetilmis Hedef Hava Ozellikleri
    final_df['DEST_FROST_RISK'] = ((final_df['DEST_TEMP'] <= 3) & ((final_df['DEST_TEMP'] - final_df['DEST_DEW_POINT']).abs() <= 3)).astype(int)
    final_df['DEST_IS_LVP'] = (final_df['DEST_VISIBILITY'] < 1500).astype(int)
    
    final_df.to_parquet("data/dest_weather.parquet", index=False)
    print("Hedef hava durumu basariyla kaydedildi!")
else:
    print("Veri indirilemedi!")

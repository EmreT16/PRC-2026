"""
Hava durumu birlestirme (kalkis ve varis)
+ etkilesim ozellikleri #78-80 (pist acisi, headwind, crosswind) ve #81
(sicaklik-cig noktasi farki).

Kaynak dosyalar (PRC_2/data/ altinda, harici Open-Meteo API'den onceden
indirilmis): weather_all.parquet (kalkis), dest_weather.parquet (varis).
Esleme anahtari: (havalimani, UTC yil, ay, gun, saat) - saatlik cozunurluk.

UYARI: varis hava durumu ozellikleri eski
projede olculup RMSE'yi kotulestirdigi (+3.21 sn) icin cikarilmisti - burada
uretiliyor ama kullanilmadan once mutlaka ayri olculmeli.
"""
import numpy as np
import pandas as pd

# Pist kodundan aci cikarma: ilk 2 rakam x 10 derece. 'H' gibi rakam
# olmayanlar (orn. LSZH helikopter pisti) NaN kalir.
def runway_heading(runway_series):
    digits = runway_series.astype(str).str.extract(r'^(\d{2})')[0]
    heading = pd.to_numeric(digits, errors='coerce') * 10
    return heading


def add_weather_and_wind_features(df, weather_dep_path, weather_dest_path):
    """df: MVT_TIME_UTC_mvt, ADEP_mvt, ADES_mvt, RUNWAY_mvt iceren kalkis
    tablosu. Donen: df ile ayni sirada, hava durumu + ruzgar ozellikleri.
    """
    t = pd.to_datetime(df['MVT_TIME_UTC_mvt'])
    key = pd.DataFrame({
        'ADEP_mvt': df['ADEP_mvt'].to_numpy(),
        'ADES_mvt': df['ADES_mvt'].to_numpy(),
        'MVT_YEAR': t.dt.year.to_numpy(),
        'MVT_MONTH': t.dt.month.to_numpy(),
        'MVT_DAY': t.dt.day.to_numpy(),
        'MVT_HOUR': t.dt.hour.to_numpy(),
    }, index=df.index)

    # --- Kalkis hava durumu (kategori 2.7, ozellik 47-66) -------------------
    w = pd.read_parquet(weather_dep_path)
    w = w.rename(columns={'CURRENT_AIRPORT': 'ADEP_mvt'})
    dep_w = key.merge(w, on=['ADEP_mvt', 'MVT_YEAR', 'MVT_MONTH', 'MVT_DAY', 'MVT_HOUR'],
                       how='left')
    dep_w.index = df.index

    # --- Varis hava durumu (kategori 2.8, ozellik 67-76) ---------------------
    dw = pd.read_parquet(weather_dest_path)
    dest_w = key.merge(dw, on=['ADES_mvt', 'MVT_YEAR', 'MVT_MONTH', 'MVT_DAY', 'MVT_HOUR'],
                        how='left')
    dest_w.index = df.index

    out = pd.concat([
        dep_w.drop(columns=['ADEP_mvt', 'ADES_mvt', 'MVT_YEAR', 'MVT_MONTH', 'MVT_DAY', 'MVT_HOUR']),
        dest_w.drop(columns=['ADEP_mvt', 'ADES_mvt', 'MVT_YEAR', 'MVT_MONTH', 'MVT_DAY', 'MVT_HOUR']),
    ], axis=1)

    # --- Etkilesim #78-80: pist acisi, headwind, crosswind -------------------
    heading = runway_heading(df['RUNWAY_mvt'])
    out['RUNWAY_HEADING_DEG'] = heading
    rad = np.deg2rad(heading)
    wind_u = out.get('WIND_U')
    wind_v = out.get('WIND_V')
    if wind_u is not None and wind_v is not None:
        # Ucagin burnu pist yonune (heading) bakiyor kabul edilir. Ruzgar
        # vektorunun bu yon uzerindeki izdusumu = headwind (pozitifse burundan),
        # dik bileseni = crosswind (mutlak deger, yon farketmez).
        out['HEADWIND'] = wind_u * np.sin(rad) + wind_v * np.cos(rad)
        out['CROSSWIND'] = np.abs(wind_u * np.cos(rad) - wind_v * np.sin(rad))
    else:
        out['HEADWIND'] = np.nan
        out['CROSSWIND'] = np.nan

    # --- Etkilesim #81: sicaklik - cig noktasi farki (surekli) ---------------
    if 'TEMPERATURE' in out.columns and 'DEW_POINT' in out.columns:
        out['TEMP_DEWPOINT_SPREAD'] = out['TEMPERATURE'] - out['DEW_POINT']
    else:
        out['TEMP_DEWPOINT_SPREAD'] = np.nan

    return out

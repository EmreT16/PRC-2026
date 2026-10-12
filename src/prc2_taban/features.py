"""
ANA OZELLIK MUHENDISLIGI MODULU.

Taban hattin 117 ozelliginin (13 kategori) tamamini uretir. Her blok,
belgedeki kategori numarasina gonderme yapan yorumla isaretlenmistir.

Kullanim: build_dataset.py bu modulu cagirir. Bu dosya dogrudan calistirilmaz.

Girdi: `raw_all` - training_*.parquet dosyalarinin TAMAMI (12 ay) + ranking.parquet
(hem DEP hem ARR satirlari) UST USTE BINDIRILMEDEN, TEK bir DataFrame'de
BIRLESTIRILMIS halde verilmelidir. Bu zorunludur, cunku trafik/kuyruk ve stand
devri ozellikleri ay sinirlarini ve egitim/test sinirini asan zaman pencereleri
kullanir (orn. 31 Aralik 23:50'deki bir kalkis icin "son 30 dk" 1 Ocak'a sarkmaz
ama Aralik dosyasinin kendi icinde kalir - ayri ayri isleseydik bu sinirlarda
yanlis/eksik sayim olurdu).
"""
import sys
import os
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
from airport_info import AIRPORT_INFO, get_country_from_icao
from traffic_features import build_traffic_features, wake_turbulence_flag, rolling_count
from group_stats import add_stand_runway_stats, add_aircraft_operator_stats
from weather_merge import add_weather_and_wind_features
from holiday_features import compute_holiday_features

TARGET = 'TAXITIME_SEC_mvt'


def _local_time(utc_series, apt_series):
    """Her satiri kendi havalimaninin saat dilimine cevirir (ozellik 26-27,
    106). Havalimani bazinda gruplanip tz_convert ile hizli yapilir.

    Onemli: sonuc NAIVE (tz'siz) datetime64 olarak tutulur - eger kapsayici
    Series bastan tz-aware (orn. UTC) tanimlanirsa, icine baska bir tz'den
    atanan degerler otomatik olarak container'in tz'sine (UTC'ye) geri
    cevrilir ve yerel saat donusumunun ETKISI SESSIZCE IPTAL OLUR. Bu yuzden
    once tz_convert (dogru yerel ana tasi), sonra tz_localize(None) (etiketi
    at, saat degerini AYNEN birak - "duvar saati" olarak) yapilir.
    """
    utc = pd.to_datetime(utc_series)
    if utc.dt.tz is None:
        utc = utc.dt.tz_localize('UTC')
    out = pd.Series(pd.NaT, index=utc.index, dtype='datetime64[ns]')
    for apt, (tz, _cc) in AIRPORT_INFO.items():
        mask = (apt_series == apt).to_numpy()
        if mask.any():
            local_wall = utc.loc[mask].dt.tz_convert(ZoneInfo(tz)).dt.tz_localize(None)
            out.loc[mask] = local_wall
    return out


def build_features(raw_all, weather_dep_path, weather_dest_path):
    df = raw_all.copy()

    # Tum zaman damgalari UTC tz-aware geliyor. Numpy/searchsorted islemleri
    # tz-aware dtype ile sorun cikardigi icin, deger degismeden (zaten UTC)
    # tz etiketi kaldirilir (naive UTC). _local_time fonksiyonu gerektiginde
    # kendi tz_localize('UTC') adimini tekrar yapar.
    for c in ['MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
              'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt']:
        if c in df.columns and pd.api.types.is_datetime64tz_dtype(df[c]):
            df[c] = df[c].dt.tz_localize(None)

    # --- Ortak alanlar (hem DEP hem ARR icin gerekli, trafik ozellikleri icin) ---
    df['APT'] = np.where(df['PHASE_mvt'] == 'DEP', df['ADEP_mvt'], df['ADES_mvt'])
    df['MVT_T'] = df['MVT_TIME_UTC_mvt']
    df['OFFBLOCK_T'] = df['AOBT_3_flt']
    df['IS_DEP'] = df['PHASE_mvt'] == 'DEP'
    df['IS_ARR'] = df['PHASE_mvt'] == 'ARR'
    df['STAND_ALL'] = df['APT'] + '_' + df['STAND_mvt'].astype(str)

    # --- Kategori 2.11 / #111: stand devri (once, DEP+ARR birlikteyken) -----
    # Ayni park yerine en son inen ucagin ne zaman geldigi ve onun taxi-in
    # suresi. Sadece geciye bakar (merge_asof backward), gelecegi gormez.
    arr_cols = ['STAND_ALL', 'MVT_T'] + ([TARGET] if TARGET in df.columns else [])
    arr_side = (df.loc[df['IS_ARR'], arr_cols]
                .dropna(subset=['STAND_ALL', 'MVT_T'])
                .sort_values('MVT_T')
                .rename(columns={'MVT_T': 'PREV_STAND_ARR_T', TARGET: 'PREV_STAND_ARR_TAXI'}))
    arr_side['_asof'] = arr_side['PREV_STAND_ARR_T']

    left = df.loc[df['IS_DEP'] & df['MVT_T'].notna() & df['STAND_ALL'].notna(),
                  ['STAND_ALL', 'MVT_T']].copy()
    left['_asof'] = left['MVT_T']
    left = left.sort_values('_asof')
    got = pd.merge_asof(left.reset_index(), arr_side, on='_asof', by='STAND_ALL',
                         direction='backward', allow_exact_matches=False)
    got = got.set_index('index')

    # --- Simdiden itibaren sadece kalkislar ----------------------------------
    d = df[df['IS_DEP']].copy()
    d['PREV_STAND_ARR_TAXI'] = got.reindex(d.index)['PREV_STAND_ARR_TAXI']
    sec_since = (d['MVT_T'] - got.reindex(d.index)['PREV_STAND_ARR_T']).dt.total_seconds()
    d['SEC_SINCE_STAND_ARR'] = sec_since.clip(upper=86400)  # #111

    # NM_TAXI_DIFF_SEC (kritik capa ozelligi) burada, trafik dongusunden ONCE
    # hesaplanir - hem #19'un kaynagi hem kategori 2.6/2.10'un girdisi.
    d['NM_TAXI_DIFF_SEC'] = (d['MVT_T'] - d['AOBT_3_flt']).dt.total_seconds()

    out = pd.DataFrame(index=d.index)

    # =========================================================================
    # KATEGORI 2.1 - Dogrudan kategori (ozellik 1-9)
    # =========================================================================
    for c in ['ADEP_mvt', 'ADES_mvt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt',
              'CALLSIGN_flt', 'MARKET_SEGMENT_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt']:
        out[c] = d[c]

    # =========================================================================
    # KATEGORI 2.2 - Metinden bolunen (ozellik 10-14)
    # =========================================================================
    out['FLIGHT_CODE'] = d['FLIGHT_mvt']                                          # #10
    flight_str = d['FLIGHT_mvt'].astype(str)
    out['AIRLINE_LETTERS'] = flight_str.str.extract(r'([A-Za-z]+)')[0]            # #11
    out['FLIGHT_NUMBER'] = pd.to_numeric(flight_str.str.extract(r'(\d+)')[0], errors='coerce')  # #12
    out['ADES_FILED_CHANGED'] = (d['ADES_FILED_flt'] != d['ADES_flt']).astype(float)  # #13
    out['ADES_FILED'] = d['ADES_FILED_flt']                                       # #14

    # =========================================================================
    # KATEGORI 2.3 - Tarih farki: diger sutun - MVT_TIME_UTC_mvt (ozellik 15-20)
    # =========================================================================
    mvt = d['MVT_T']
    out['SCHED_DIFF_SEC'] = (d['SCHED_TIME_UTC_mvt'] - mvt).dt.total_seconds()    # #15
    out['LOBT_DIFF_SEC'] = (d['LOBT_flt'] - mvt).dt.total_seconds()               # #16
    out['IOBT_DIFF_SEC'] = (d['IOBT_flt'] - mvt).dt.total_seconds()               # #17
    out['EOBT_DIFF_SEC'] = (d['EOBT_1_flt'] - mvt).dt.total_seconds()             # #18
    out['AOBT_DIFF_SEC'] = (d['AOBT_3_flt'] - mvt).dt.total_seconds()             # #19
    out['ARVT3_DIFF_SEC'] = (d['ARVT_3_flt'] - mvt).dt.total_seconds()            # #20
    # Yeni: NM'nin AOBT tahmini ile daha erken EOBT tahmini arasindaki fark -
    # Model A'da CV RMSE 228.32->223.39 (5 tohum, std 0.10, kararli).
    # Feature importance taramasinda bulundu.
    out['AOBT_MINUS_EOBT'] = out['AOBT_DIFF_SEC'] - out['EOBT_DIFF_SEC']
    out['NM_TAXI_DIFF_SEC'] = d['NM_TAXI_DIFF_SEC']
    nm_taxi_diff = d['NM_TAXI_DIFF_SEC']

    # =========================================================================
    # KATEGORI 2.4 - UTC zaman bilgisi (ozellik 21-25)
    # =========================================================================
    out['MVT_MONTH'] = mvt.dt.month                                               # #21
    out['MVT_DAY'] = mvt.dt.day                                                   # #22
    out['MVT_DAYOFWEEK'] = mvt.dt.dayofweek                                       # #23
    out['MVT_HOUR_UTC'] = mvt.dt.hour                                             # #24
    out['MVT_MINOFDAY_UTC'] = mvt.dt.hour * 60 + mvt.dt.minute                    # #25

    # =========================================================================
    # KATEGORI 2.5 - Yerel saat (ozellik 26-27) + 2.11 dongusel yerel saat (#106)
    # =========================================================================
    local_t = _local_time(mvt, d['ADEP_mvt'])
    out['LOCAL_HOUR'] = local_t.dt.hour                                           # #26
    local_minofday = local_t.dt.hour * 60 + local_t.dt.minute
    out['LOCAL_MINOFDAY'] = local_minofday                                        # #27
    out['HOUR_SIN'] = np.sin(2 * np.pi * local_minofday / 1440)                   # #106
    out['HOUR_COS'] = np.cos(2 * np.pi * local_minofday / 1440)                   # #106
    out['IS_PEAK_HOUR'] = out['LOCAL_HOUR'].isin([6, 7, 8, 9, 16, 17, 18, 19]).astype(float)  # #107

    # =========================================================================
    # KATEGORI 2.6 - Trafik/kuyruk (ozellik 28-46) + yeni #114 (bu icinde)
    # =========================================================================
    traffic_frames = []
    wake_frames = []
    for apt in sorted(d['ADEP_mvt'].dropna().unique()):
        dep_i = d[d['ADEP_mvt'] == apt]
        arr_i = df[(df['IS_ARR']) & (df['APT'] == apt)]
        tf = build_traffic_features(dep_i, arr_i)
        traffic_frames.append(tf)
        wake_frames.append(wake_turbulence_flag(dep_i))
    traffic_all = pd.concat(traffic_frames).reindex(d.index)
    wake_all = pd.concat(wake_frames).reindex(d.index)
    for c in traffic_all.columns:
        out[c] = traffic_all[c]
    out['WAKE_TURBULENCE_BEHIND_HEAVY'] = wake_all                                 # #83

    # --- Yeni #117 (basitlestirilmis): son 30 dk'da aktif pist sayisi --------
    # Not: ilk tasarimda "degisim" olarak dusunulmustu (dusuk oncelik,
    # karmasik). Burada daha basit ama yine bilgi tasiyan bir vekil kullanildi:
    # dogrudan "su an kac farkli pist aktif" sayisi (degisim tespiti degil).
    col = np.full(len(d), np.nan)
    for apt in sorted(d['ADEP_mvt'].dropna().unique()):
        mask = (d['ADEP_mvt'] == apt).to_numpy()
        sub = d[d['ADEP_mvt'] == apt]
        times = sub['MVT_T'].to_numpy()
        rwy = sub['RUNWAY_mvt'].to_numpy()
        vals = np.full(len(sub), np.nan)
        order = np.argsort(times)
        times_sorted = times[order]
        rwy_sorted = rwy[order]
        window = np.timedelta64(30, 'm')
        lo = np.searchsorted(times_sorted, times_sorted - window, side='left')
        hi = np.searchsorted(times_sorted, times_sorted, side='left')
        for i in range(len(times_sorted)):
            vals[order[i]] = len(set(rwy_sorted[lo[i]:hi[i]])) if hi[i] > lo[i] else 0
        col[mask] = vals
    out['ACTIVE_RUNWAY_COUNT_30'] = col                                            # #117

    # =========================================================================
    # KATEGORI 2.7 / 2.8 / etkilesim 78-81 - Hava durumu + ruzgar
    # =========================================================================
    weather = add_weather_and_wind_features(d, weather_dep_path, weather_dest_path)
    for c in weather.columns:
        out[c] = weather[c]

    # =========================================================================
    # KATEGORI 2.9 - Resmi tatiller (ozellik 88-101)
    # =========================================================================
    dep_country = d['ADEP_mvt'].map(lambda a: AIRPORT_INFO.get(a, (None, None))[1])
    dep_hol = compute_holiday_features(local_t, dep_country, 'dep')
    dep_hol.index = d.index
    for c in dep_hol.columns:
        out[c] = dep_hol[c]

    ades_country = get_country_from_icao(d['ADES_mvt'])
    ades_hol = compute_holiday_features(mvt, ades_country, 'ades')
    ades_hol.index = d.index
    for c in ades_hol.columns:
        out[c] = ades_hol[c]

    # =========================================================================
    # KATEGORI 2.10 - Grup gecmis istatistikleri (ozellik 102-105, 112-113)
    # =========================================================================
    d_for_groups = d.assign(NM_TAXI_DIFF_SEC=nm_taxi_diff)
    stand_stats = add_stand_runway_stats(d_for_groups)
    for c in stand_stats.columns:
        out[c] = stand_stats[c]
    op_stats = add_aircraft_operator_stats(d_for_groups)
    for c in op_stats.columns:
        out[c] = op_stats[c]

    # =========================================================================
    # KATEGORI 2.11 - Eski projeden bulunan ek ozellikler (ozellik 108-111)
    # =========================================================================
    out['NM_MISSING'] = d['AOBT_3_flt'].isna().astype(float)                       # #108
    takeoff_vs_sched = (mvt - d['SCHED_TIME_UTC_mvt']).dt.total_seconds()
    out['TAKEOFF_VS_SCHED_SEC'] = takeoff_vs_sched
    out['TAKEOFF_VS_SCHED_SEC_POS'] = takeoff_vs_sched.clip(lower=0)               # #109
    out['TAKEOFF_VS_SCHED_SEC_NEG'] = takeoff_vs_sched.clip(upper=0)               # #110
    out['PREV_STAND_ARR_TAXI'] = d['PREV_STAND_ARR_TAXI']                          # #111
    out['SEC_SINCE_STAND_ARR'] = d['SEC_SINCE_STAND_ARR']                          # #111

    # =========================================================================
    # KATEGORI 2.9(etkilesim)/2.12 - Kalan turetilmis ve yeni ozellikler
    # =========================================================================
    out['OPERATIONAL_DIFFICULTY'] = out['SURFACE_N'] * out['PRECIPITATION'].fillna(0)  # #77
    out['DELAY_QUEUE_PRESSURE'] = out['TAKEOFF_VS_SCHED_SEC_POS'] * out['DEP_QUEUE_N']  # #82
    out['STAND_TYPE'] = d['STAND_mvt'].astype(str).str[0]                          # #84
    out['ROUTE_PAIR'] = d['ADEP_mvt'] + '_' + d['ADES_mvt']                        # #87

    dep_cc = d['ADEP_mvt'].map(lambda a: AIRPORT_INFO.get(a, (None, None))[1])
    ades_cc2 = get_country_from_icao(d['ADES_mvt'])
    out['IS_DOMESTIC'] = (dep_cc.to_numpy() == ades_cc2.to_numpy()).astype(float)   # #115

    # #116: gunun kacinci kalkisi (havalimani bazinda, o gunun basindan beri)
    day_key = d['ADEP_mvt'] + '_' + mvt.dt.date.astype(str)
    rank_in_day = d.assign(_day=day_key, _t=mvt).sort_values('_t').groupby('_day').cumcount() + 1
    out['DEP_RANK_TODAY'] = rank_in_day.reindex(d.index)
    out['IS_FIRST_DEP_TODAY'] = (out['DEP_RANK_TODAY'] == 1).astype(float)          # #116

    # --- saat dilimi bazli oran ----------------------------------------------
    # DEP_QUEUE_N'i mutlak degil, o havalimaninin o saatteki TARIHSEL ortalamasina
    # oranla veriyoruz - hedeften turemedigi (girdi ozelligi) icin sizinti degil,
    # train+test birlikte hesaplaniyor (yapisal bir trafik deseni).
    hour_mean = out.groupby([d['ADEP_mvt'], out['LOCAL_HOUR']])['DEP_QUEUE_N'].transform('mean')
    out['QUEUE_RATIO_TO_HOUR_MEAN'] = out['DEP_QUEUE_N'] / (hour_mean + 1)

    # --- yuzdelik siralama ---------------------------------------------------
    # Bu ucusun rotari (SCHED_DIFF_SEC), ayni gun+saat+havalimaninda kalkan diger
    # ucuslara kiyasla yuzde kacinci dilimde? Hedeften turemez, sizinti yok.
    out['SCHED_DIFF_SEC_HOUR_PCT'] = out.groupby([day_key, out['LOCAL_HOUR']])['SCHED_DIFF_SEC'].rank(pct=True)

    # --- Hedef ve kimlik (sona eklenir, ozellik listesine dahil degil ama
    #     egitim/degerlendirme icin gerekli) --------------------------------
    out['MVT_ID_mvt'] = d['MVT_ID_mvt']
    if TARGET in d.columns:
        out[TARGET] = d[TARGET]

    # --- NM_OFFSET_SEC: SADECE EGITIMDE var olan ikinci hedef (Model A icin) ---
    # BLOCK_TIME_UTC_mvt test setinde (rank) tamamen bos oldugundan bu sutun
    # test satirlarinda otomatik olarak NaN cikar - ayrica ozel kod gerekmez.
    # DIKKAT: bu bir GIRDI OZELLIGI DEGILDIR, hedeften (BLOCK_TIME) turedigi
    # icin FEATURES listesine asla konmaz - sadece Model A'nin egitim hedefi
    # olarak kullanilir (KARARLAR_VE_YAPILANLAR.md bolum 4).
    if 'BLOCK_TIME_UTC_mvt' in d.columns:
        out['NM_OFFSET_SEC'] = (d['BLOCK_TIME_UTC_mvt'] - d['AOBT_3_flt']).dt.total_seconds()

    return out.reset_index(drop=True)

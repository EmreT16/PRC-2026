"""
Trafik/kuyruk ozellikleri +
etkilesim ozelliklerinden #83 (wake turbulence) ve #117 (aktif pist degisimi) +
yeni ozellik #114 (yakin gecmis kalkis taksi egilimi).

Tum fonksiyonlar TEK bir havalimani grubu icin calisir ve caginan yerde
havalimanlari uzerinde donguyle (sadece 10 havalimani var, hizli) cagirilir.

Onemli: hicbir fonksiyon "gelecekten" bilgi kullanmaz - her zaman SORGU ZAMANINDAN
KESINLIKLE ONCEKI olaylar sayilir/kullanilir (strict-before).
"""
import numpy as np
import pandas as pd


def rolling_count(event_times, query_times, window_minutes):
    """event_times (sirali, ayni grup) icinde her query_times[i] icin,
    (query_times[i] - window, query_times[i]) araliginda (query anini haric
    tutarak) kac event_times oldugunu sayar. Vektorize (searchsorted).
    """
    ev = np.sort(event_times[~pd.isna(event_times)].astype('datetime64[ns]'))
    q = query_times.astype('datetime64[ns]')
    window = np.timedelta64(window_minutes, 'm')
    lo = np.searchsorted(ev, q - window, side='left')
    hi = np.searchsorted(ev, q, side='left')  # kesinlikle once
    return (hi - lo).astype(float)


def rolling_stat_before(event_times, event_values, query_times, window_minutes, stat='median'):
    """event_times/event_values (ayni grup) icinde her query zamani icin,
    query'den ONCEKI window_minutes penceresindeki event_values uzerinde
    median/max hesaplar. Pandas'in zaman-tabanli rolling'i icin, event ve
    query zaman damgalarini TEK bir sirali eksende birlestirip (event'lere
    gercek deger, query'lere NaN vererek) rolling(window, closed='left')
    uyguluyoruz - boylece query noktasinda sadece ONCEKI event degerleri
    hesaba katilir (pandas nan-farkinda oldugu icin NaN'lar otomatik atlanir).
    """
    ev_mask = ~pd.isna(event_times)
    ev_t = pd.to_datetime(event_times[ev_mask])
    ev_v = np.asarray(event_values)[ev_mask]

    q_t = pd.to_datetime(query_times)

    combo = pd.DataFrame({
        'val': np.concatenate([ev_v, np.full(len(q_t), np.nan)]),
        'is_query': np.concatenate([np.zeros(len(ev_v), dtype=bool), np.ones(len(q_t), dtype=bool)]),
    }, index=pd.DatetimeIndex(pd.concat([pd.Series(ev_t), pd.Series(q_t)], ignore_index=True)))
    combo = combo.sort_index()

    win = f'{window_minutes}min'
    if stat == 'median':
        res = combo['val'].rolling(win, closed='left').median()
    elif stat == 'max':
        res = combo['val'].rolling(win, closed='left').max()
    elif stat == 'sum':
        res = combo['val'].rolling(win, closed='left').sum()
    elif stat == 'p90':
        res = combo['val'].rolling(win, closed='left').quantile(0.9)
    else:
        raise ValueError(stat)

    return res[combo['is_query'].to_numpy()].to_numpy()


def concurrent_count(start_times, end_times, query_times, include_self=True):
    """Her query_times[i] icin, [start,end) araligi query anini iceren kac
    kayit oldugunu sayar (ayni gruptaki tum start/end ciftleri arasindan).
    Olay-farki (event difference) + kumulatif toplam yontemiyle O(n log n).

    Veri kalitesi notu: bazi satirlarda (AOBT kayit hatasi nedeniyle) start
    > end olabiliyor (kayittaki nadir anomaliler - bkz. NM_TAXI_DIFF_SEC'in
    bazen negatif cikmasi). Boyle bozuk araliklar, olay zaman cizelgesini
    PAYLASTIKLARI icin baska uculardaki sayimi da negatife dusurebiliyor.
    Bu yuzden once start<=end olan gecerli araliklar filtrelenir; ayrica
    sonuc guvenlik icin 0'da alt sinira kirpilir (sayim asla eksi olamaz).
    """
    mask = ~(pd.isna(start_times) | pd.isna(end_times))
    s = pd.to_datetime(np.asarray(start_times)[mask]).to_numpy('datetime64[ns]')
    e = pd.to_datetime(np.asarray(end_times)[mask]).to_numpy('datetime64[ns]')

    valid = s <= e
    s, e = s[valid], e[valid]

    if len(s) == 0:
        return np.zeros(len(query_times))

    ev_time = np.concatenate([s, e])
    ev_delta = np.concatenate([np.ones(len(s)), -np.ones(len(e))])
    order = np.argsort(ev_time, kind='stable')
    ev_time = ev_time[order]
    ev_delta = ev_delta[order]
    cum = np.cumsum(ev_delta)

    q = pd.to_datetime(query_times).to_numpy('datetime64[ns]')
    # include_self=True: kendi start olayi da dahil olsun diye side='right'
    side = 'right' if include_self else 'left'
    idx = np.searchsorted(ev_time, q, side=side) - 1
    out = np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)
    return np.clip(out, 0, None)  # guvenlik: sayim asla eksi olamaz


def build_traffic_features(dep, arr):
    """dep: DEP satirlari (bir havalimani icin), arr: ayni havalimaninin ARR
    satirlari. dep ile ayni sirada, kategori 2.6'daki 14 ozelligi (+ #114)
    iceren bir DataFrame dondurur.

    Gerekli kolonlar:
      dep: MVT_T (kalkis ani), OFFBLOCK_T (AOBT_3_flt), RUNWAY_mvt, NM_TAXI_DIFF_SEC
      arr: MVT_T (inis ani), TAXITIME_SEC_mvt (taxi-in suresi)
    """
    out = pd.DataFrame(index=dep.index)
    dep_mvt = dep['MVT_T'].to_numpy()
    dep_off = dep['OFFBLOCK_T'].to_numpy()

    # --- Q_PUSHBACK_30/60: ayni havalimaninda pushback yapan kalkis sayisi ---
    for w in (30, 60):
        out[f'Q_PUSHBACK_{w}'] = rolling_count(dep_off, dep_off, w)

    # --- Q_RWY_DEP_15/30/60: ayni PISTTEN kalkan ucus sayisi -----------------
    for w in (15, 30, 60):
        col = np.full(len(dep), np.nan)
        for rwy, idx in dep.groupby('RUNWAY_mvt').groups.items():
            pos = dep.index.get_indexer(idx)
            times = dep_mvt[pos]
            col[pos] = rolling_count(times, times, w)
        out[f'Q_RWY_DEP_{w}'] = col

    # --- Q_ARR_15/30/60: ayni havalimanina inen ucus sayisi ------------------
    arr_mvt = arr['MVT_T'].to_numpy()
    for w in (15, 30, 60):
        out[f'Q_ARR_{w}'] = rolling_count(arr_mvt, dep_mvt, w)

    # --- ARR_TAXI_MED_30/60, ARR_TAXI_MAX_30: varis taksi istatistigi -------
    arr_taxi = arr['TAXITIME_SEC_mvt'].to_numpy()
    out['ARR_TAXI_MED_30'] = rolling_stat_before(arr_mvt, arr_taxi, dep_mvt, 30, 'median')
    out['ARR_TAXI_MED_60'] = rolling_stat_before(arr_mvt, arr_taxi, dep_mvt, 60, 'median')
    out['ARR_TAXI_MAX_30'] = rolling_stat_before(arr_mvt, arr_taxi, dep_mvt, 30, 'max')

    # --- Yeni #114: yakin gecmis KALKIS taksi egilimi (NM_TAXI_DIFF_SEC) ----
    dep_nm = dep['NM_TAXI_DIFF_SEC'].to_numpy()
    out['DEP_TAXI_MED_15'] = rolling_stat_before(dep_mvt, dep_nm, dep_mvt, 15, 'median')
    out['DEP_TAXI_MED_30'] = rolling_stat_before(dep_mvt, dep_nm, dep_mvt, 30, 'median')

    # --- Anlik kuyruk uzunlugu (concurrent count) ----------------------------
    dep_end = dep_mvt  # kalkis: pushback ile takeoff arasi "yolda"
    out['DEP_QUEUE_N'] = concurrent_count(dep_off, dep_end, dep_mvt, include_self=True)

    col = np.full(len(dep), np.nan)
    for rwy, idx in dep.groupby('RUNWAY_mvt').groups.items():
        pos = dep.index.get_indexer(idx)
        col[pos] = concurrent_count(dep_off[pos], dep_end[pos], dep_mvt[pos], include_self=True)
    out['RWY_QUEUE_N'] = col

    arr_taxi_sec = pd.to_timedelta(arr['TAXITIME_SEC_mvt'].fillna(0), unit='s')
    arr_start = arr_mvt
    arr_end = (pd.to_datetime(arr['MVT_T']) + arr_taxi_sec).to_numpy()
    out['ARR_TAXIING_N'] = concurrent_count(arr_start, arr_end, dep_mvt, include_self=False)
    out['SURFACE_N'] = out['DEP_QUEUE_N'].fillna(0) + out['ARR_TAXIING_N'].fillna(0)

    # --- DEP_15 / ARR_15 / DEP_ARR_RATIO / MOVEMENTS_15 -----------------------
    out['DEP_15'] = rolling_count(dep_mvt, dep_mvt, 15)
    out['ARR_15'] = rolling_count(arr_mvt, dep_mvt, 15)
    out['DEP_ARR_RATIO'] = out['DEP_15'] / (out['ARR_15'] + 1)
    out['MOVEMENTS_15'] = out['DEP_15'] + out['ARR_15']

    return out


def wake_turbulence_flag(dep):
    """Ozellik #83: aynı pistte kendisinden HEMEN ONCEKI kalkisin agirlik
    kategorisi Heavy/Jumbo mu (0/1). merge_asof (backward) ile - gelecege
    bakmaz.
    """
    d = dep[['MVT_T', 'RUNWAY_mvt', 'WK_TBL_CAT_flt']].copy()
    d = d.sort_values('MVT_T')
    d['_prev_wk'] = d.groupby('RUNWAY_mvt')['WK_TBL_CAT_flt'].shift(1)
    flag = d['_prev_wk'].isin(['H', 'J']).astype(float)
    return flag.reindex(dep.index)

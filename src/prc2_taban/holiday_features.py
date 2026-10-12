"""
Resmi tatil ozellikleri.

Kaynak: Python `holidays` kutuphanesi (acik kaynak, ucretsiz, ulke bazli resmi
tatil takvimleri).

7 ozellik uretilir, hem kalkis (dep_) hem varis (ades_) tarafi icin ayri ayri
cagirilir:
  - is_holiday_today          (0/1)
  - days_until_holiday        (0-15, sabitlenmis)
  - holiday_name               (kategorik - yaklasan/mevcut tatilin adi/sinifi)
  - holiday_duration            (o tatil bloğunun toplam gun sayisi)
  - days_until_holiday_end      (tatil icindeyse bitmesine kalan gun)
  - days_since_holiday_end      (0-15, sabitlenmis - biten tatilden beri gecen gun)
  - prev_holiday_name           (kategorik - en son biten tatilin adi/sinifi)

Onemli tasarim karari: "bugun tatil mi" bayragi COK GUNLU
tatillerin ortasini dogru yakalamak icin GEREKLIDIR, sadece "days_until_holiday==0"
ile degistirilemez.

Performans notu: milyonlarca satirin cogu ayni (ulke, tarih) ciftini paylasir
(orn. ayni gun ayni havalimanindan kalkan yuzlerce ucus). Bu yuzden once TEKIL
(ulke, tarih) ciftleri icin hesaplama yapilir, sonra tum satirlara geri birlestirilir
(merge) - milyonlarca satirda tek tek Python dongusu calistirmaktan cok daha hizli.
"""
import numpy as np
import pandas as pd
import holidays as pyholidays

CAP_DAYS = 15  # "en fazla 15" karari

_CAL_CACHE = {}


def _country_holiday_blocks(country_code, years):
    """Bir ulke icin ardisik tatil+hafta sonu bloklarini hesaplar (cache'li).

    Donen: [(baslangic_tarih, bitis_tarih, ad), ...] tarihe gore sirali liste.
    """
    key = (country_code, tuple(sorted(years)))
    if key in _CAL_CACHE:
        return _CAL_CACHE[key]
    try:
        yrs = sorted(set(years) | {min(years) - 1, max(years) + 1})
        cal = pyholidays.country_holidays(country_code, years=yrs)
    except Exception:
        _CAL_CACHE[key] = []
        return []
    blocks = _blocks_from_calendar(cal)
    _CAL_CACHE[key] = blocks
    return blocks


def _blocks_from_calendar(cal):
    if cal is None or len(cal) == 0:
        return []
    dates = sorted(pd.Timestamp(d) for d in cal.keys())
    blocks = []
    i = 0
    while i < len(dates):
        start = dates[i]
        name = cal.get(start.date(), 'Holiday')
        cur_end = start
        j = i + 1
        while j < len(dates):
            gap = (dates[j] - cur_end).days
            if gap == 1 or (gap <= 3 and _is_weekend_bridge(cur_end, dates[j])):
                cur_end = dates[j]
                j += 1
            else:
                break
        blocks.append((start, cur_end, str(name)))
        i = j
    return blocks


def _is_weekend_bridge(d1, d2):
    cur = d1 + pd.Timedelta(days=1)
    while cur < d2:
        if cur.weekday() < 5:
            return False
        cur += pd.Timedelta(days=1)
    return True


def _features_for_unique_pairs(unique_df, prefix):
    """unique_df: kolonlari ['cc', 'date'] olan tekil (ulke, tarih) ciftleri.

    Donen: unique_df ile ayni sirada, 7 yeni kolon eklenmis DataFrame.
    """
    n = len(unique_df)
    is_hol = np.zeros(n)
    days_until = np.full(n, np.nan)
    hol_name = np.full(n, 'NONE', dtype=object)
    hol_dur = np.full(n, np.nan)
    days_until_end = np.full(n, np.nan)
    days_since_end = np.full(n, np.nan)
    prev_name = np.full(n, 'NONE', dtype=object)

    years_all = sorted(unique_df['date'].dt.year.dropna().unique().astype(int))

    for cc, grp in unique_df.groupby('cc', dropna=True):
        blocks = _country_holiday_blocks(cc, years_all)
        if not blocks:
            continue
        starts = pd.DatetimeIndex([b[0] for b in blocks])
        ends = pd.DatetimeIndex([b[1] for b in blocks])
        names = [b[2] for b in blocks]

        for row_i in grp.index:
            d = unique_df.at[row_i, 'date']
            in_block = (starts <= d) & (d <= ends)
            if in_block.any():
                bi = np.where(in_block)[0][0]
                is_hol[row_i] = 1.0
                days_until[row_i] = 0
                hol_name[row_i] = names[bi]
                hol_dur[row_i] = (ends[bi] - starts[bi]).days + 1
                days_until_end[row_i] = (ends[bi] - d).days
            else:
                future_mask = starts > d
                if future_mask.any():
                    fi = np.where(future_mask)[0]
                    nearest = fi[np.argmin(starts[fi] - d)]
                    days_until[row_i] = (starts[nearest] - d).days
                    hol_name[row_i] = names[nearest]
                    hol_dur[row_i] = (ends[nearest] - starts[nearest]).days + 1

            past_mask = ends < d
            if past_mask.any():
                pi = np.where(past_mask)[0]
                nearest_p = pi[np.argmax(ends[pi])]
                days_since_end[row_i] = (d - ends[nearest_p]).days
                prev_name[row_i] = names[nearest_p]

    # ONEMLI DUZELTME: tatil UZAKSA (CAP_DAYS'ten uzaksa veya hic yoksa) adi da
    # 'NONE' olmali - mesafe zaten "yakin degil" dedikten sonra HANGI tatil
    # oldugunun modele gosterilmesi anlamsiz gurultu ekler (bulundu: duzeltmeden
    # once yaz ortasindaki satirlar bile "Christmas Day" etiketi tasiyordu).
    far_or_none = np.isnan(days_until) | (days_until > CAP_DAYS)
    hol_name = np.where(far_or_none, 'NONE', hol_name)
    hol_dur = np.where(far_or_none, np.nan, hol_dur)

    far_or_none_prev = np.isnan(days_since_end) | (days_since_end > CAP_DAYS)
    prev_name = np.where(far_or_none_prev, 'NONE', prev_name)

    days_until = np.where(np.isnan(days_until), CAP_DAYS, np.clip(days_until, 0, CAP_DAYS))
    days_since_end = np.where(np.isnan(days_since_end), CAP_DAYS, np.clip(days_since_end, 0, CAP_DAYS))

    out = unique_df.copy()
    out[f'{prefix}_is_holiday_today'] = is_hol
    out[f'{prefix}_days_until_holiday'] = days_until
    out[f'{prefix}_holiday_name'] = hol_name
    out[f'{prefix}_holiday_duration'] = hol_dur
    out[f'{prefix}_days_until_holiday_end'] = days_until_end
    out[f'{prefix}_days_since_holiday_end'] = days_since_end
    out[f'{prefix}_prev_holiday_name'] = prev_name
    return out


def compute_holiday_features(dates, country_codes, prefix):
    """Verilen tarih ve ulke kodu serilerinden 7 tatil ozelligi uretir.

    dates: pd.Series (datetime) - satir sayisi kadar, orijinal sirada.
    country_codes: pd.Series (str, ISO ulke kodu, orn 'DE') - NaN olabilir.
    prefix: uretilecek kolon adlarinin oneki, orn 'dep' veya 'ades'.

    Donen: pd.DataFrame, orijinal satir sirasiyla, 7 kolonlu.
    """
    norm_dates = pd.to_datetime(dates).dt.normalize().reset_index(drop=True)
    cc = country_codes.reset_index(drop=True)

    key_df = pd.DataFrame({'cc': cc, 'date': norm_dates})
    unique_pairs = key_df.drop_duplicates().reset_index(drop=True)

    feat_unique = _features_for_unique_pairs(unique_pairs, prefix)

    merged = key_df.merge(feat_unique, on=['cc', 'date'], how='left')
    cols = [c for c in merged.columns if c.startswith(f'{prefix}_')]
    return merged[cols].reset_index(drop=True)

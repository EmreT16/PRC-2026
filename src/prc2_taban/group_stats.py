"""
Grup gecmis istatistikleri
+ yeni ozellikler #112, #113 (ucak tipi / havayolu gecmis medyani).

ONEMLI - SIZINTI KORUMASI: Tum istatistikler HEDEF (TAXITIME_SEC_mvt) yerine
NM_TAXI_DIFF_SEC uzerinden hesaplanir. NM_TAXI_DIFF_SEC hem egitimde hem test
setinde esit sekilde ve gizli hedefe hic dokunmadan mevcut oldugu icin, bu
istatistikler TUM veriden (egitim+test'in kendi NM degerleri dahil) guvenle
hesaplanabilir - klasik leave-one-out/fold kurallari gerekmez.

Fonksiyonlar tum veri (egitim+test birlikte) uzerinde CAGRILMALIDIR ki test
setindeki NM degerleri de istatistige katkida bulunsun (bu sizinti degildir,
cunku NM_TAXI_DIFF_SEC zaten aciktir).
"""
import numpy as np
import pandas as pd

MIN_COUNT = 20  # bu esigin altindaki gruplar havalimani tabanina duser


def _group_median_p10(df, group_cols, value_col='NM_TAXI_DIFF_SEC'):
    g = df.groupby(group_cols)[value_col]
    median = g.transform('median')
    p10 = g.transform(lambda x: x.quantile(0.10))
    count = g.transform('count')
    return median, p10, count


def add_stand_runway_stats(df):
    """Ozellik 102-105: (STAND_mvt, RUNWAY_mvt) grubu, yetersizse ADEP_mvt'ye
    duser.
    """
    key = df['ADEP_mvt'] + '_' + df['STAND_mvt'].astype(str) + '_' + df['RUNWAY_mvt'].astype(str)
    median, p10, count = _group_median_p10(df.assign(_key=key), ['_key'])

    apt_median = df.groupby('ADEP_mvt')['NM_TAXI_DIFF_SEC'].transform('median')
    apt_p10 = df.groupby('ADEP_mvt')['NM_TAXI_DIFF_SEC'].transform(lambda x: x.quantile(0.10))

    out = pd.DataFrame(index=df.index)
    reliable = count >= MIN_COUNT
    out['BASE_TAXI_TIME'] = np.where(reliable, median, apt_median)          # #102
    out['STAND_RWY_COUNT'] = count                                          # #103
    out['UNIMPEDED_TAXI'] = np.where(reliable, p10, apt_p10)                # #104
    out['QUEUE_DELAY_EST'] = df['NM_TAXI_DIFF_SEC'] - out['UNIMPEDED_TAXI']  # #105
    return out


def add_aircraft_operator_stats(df):
    """Ozellik #112 (ucak tipi) ve #113 (havayolu x havalimani) gecmis
    medyan taksi suresi.
    """
    out = pd.DataFrame(index=df.index)

    med_type, _, cnt_type = _group_median_p10(df, ['AIRCRAFT_TYPE_mvt'])
    apt_median = df.groupby('ADEP_mvt')['NM_TAXI_DIFF_SEC'].transform('median')
    out['AIRCRAFT_TYPE_MED_TAXI'] = np.where(cnt_type >= MIN_COUNT, med_type, apt_median)  # #112

    key = df['AIRCRAFT_OPERATOR_flt'].astype(str) + '_' + df['ADEP_mvt']
    med_op, _, cnt_op = _group_median_p10(df.assign(_key=key), ['_key'])
    out['OPERATOR_APT_MED_TAXI'] = np.where(cnt_op >= MIN_COUNT, med_op, apt_median)        # #113

    return out

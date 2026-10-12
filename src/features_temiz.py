"""C fazı: temizlenmiş özellik kümesi. v126 hattının (features_v126) üzerine kurulur.

C1: ADES_mvt, MARKET_SEGMENT_flt ve STAND_TYPE gerçek kategorik (v126'da metin → 0 hatasıyla ölüydü).
    ADES_mvt'nin eğitimde RARE_MIN'den az geçen ya da hiç görülmeyen değerleri OTHER.
C2: Kimlikler ve MVT_DAY atılır.
C3: Tüm Model A eğitim satırlarında (2.062.577) sabit ya da birebir ikiz olduğu doğrulanan sütunlar atılır (28.09):
    NM_MISSING (sabit), SCHED_DIFF_SEC (= −TAKEOFF_VS_SCHED_SEC), AOBT_MINUS_EOBT_v3 (= AOBT_MINUS_EOBT),
    EOBT_AOBT_DIFF (= −AOBT_MINUS_EOBT). LIRF_NOISY_STAND, LIRF_NOISY_16R ve ARR_15 planın ilk listesinin aksine
    sabit ya da ikiz değil (ilk 200.000 satırlık ölçüm yanıltmıştı); tutulur.
Eksik değerler 0 ile doldurulmaz; ±sonsuz NaN yapılır.
"""
import numpy as np
import pandas as pd

from . import features_v126 as F

C1_CATS = ["ADES_mvt", "MARKET_SEGMENT_flt", "STAND_TYPE"]
C2_DROP = ["CALLSIGN_flt", "FLIGHT_CODE", "AIRLINE_LETTERS", "ADES_FILED", "ROUTE_PAIR", "FLIGHT_NUMBER", "MVT_DAY"]
C3_DROP = ["NM_MISSING", "SCHED_DIFF_SEC", "AOBT_MINUS_EOBT_v3", "EOBT_AOBT_DIFF"]
CATS = F.CAT_V126 + C1_CATS
RARE_MIN = 50


def feature_list(base_columns, te=True):
    """v126 listesi − C2 − C3; te=False ise TE sütunları da atılır (C4)."""
    drop = set(C2_DROP + C3_DROP) | (set() if te else set(F.TE_COLS))
    return [c for c in F.feature_list(base_columns) if c not in drop]


def _txt(s):
    return s.fillna("nan").astype(str)


def _num(s):
    return pd.to_numeric(s, errors="coerce").astype(np.float32).replace([np.inf, -np.inf], np.nan)


def matrices(train_df, apply_df, feats):
    """Eğitim ve uygulama matrisleri. Kategoriler yalnızca eğitimden; uygulamada görülmemiş kategori NaN
    (ADES_mvt'de OTHER)."""
    X_tr, X_ap = train_df[feats].copy(), apply_df[feats].copy()
    for c in feats:
        if c in CATS:
            s_tr, s_ap = _txt(X_tr[c]), _txt(X_ap[c])
            if c == "ADES_mvt":
                vc = s_tr.value_counts()
                ok = set(vc.index[vc >= RARE_MIN])
                s_tr, s_ap = s_tr.where(s_tr.isin(ok), "OTHER"), s_ap.where(s_ap.isin(ok), "OTHER")
            cats = sorted(s_tr.unique())
            X_tr[c] = pd.Categorical(s_tr, categories=cats)
            X_ap[c] = pd.Categorical(s_ap, categories=cats)
        else:
            X_tr[c], X_ap[c] = _num(X_tr[c]), _num(X_ap[c])
    return X_tr, X_ap

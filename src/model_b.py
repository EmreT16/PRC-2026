"""Model B (NM'siz satırlar, test 5.290) — bu deponun sürümü. Hedef doğrudan taksi (TAXITIME_SEC_mvt).

Veri (29.09 ölçümü, eğitim 22.470 satır): hedef varyansının %45'i yer tutucu satırlarda (BLOCK = SCHED; 717 satır,
686'sı LIRF), %46'sı 24 saatlik tarih kaymasında (15 satır, 13'ü LIRF). Normal satırlar (%93) varyansın %3'ü.

Karışım: tahmin = p_yt·max(60, GECIKME) + p_24·D24 + p_n·μ_n
- GECIKME = MVT − SCHED (TAKEOFF_VS_SCHED_SEC); yer tutucuda taksi tam olarak bu.
- D24 = GECIKME > 40.000 ise GECIKME, değilse 86.400 + eğitimdeki 24 saat satırlarının (hedef − 86.400) medyanı.
- p: çok sınıflı LightGBM (normal / yer tutucu / 24 saat), tüm havalimanları birlikte (ADEP kategorik).
- μ_n: normal ve "uzun diğer" satırlarla (yer tutucu ve 24 saat dışı) LightGBM regresyon; havalimanı bazlı ya da global.
Rejimler: yer tutucu = \\|taksi − GECIKME\\| ≤ 60 ve taksi ≥ 2.000; 24 saat = taksi ≥ 80.000.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import config, features_temiz as FT, features_v126 as F

TARGET_B = "TAXITIME_SEC_mvt"
DELAY = "TAKEOFF_VS_SCHED_SEC"
CATS_B = ["ADEP_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "STAND_TYPE", "ADES_mvt", "WK_TBL_CAT_flt"]
DROP_B = set(F.TE_COLS) | set(FT.C2_DROP) | set(FT.C3_DROP) | {"CALLSIGN_flt", "MARKET_SEGMENT_flt",
                                                                   "AIRCRAFT_OPERATOR_flt"}
CLF_B = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=31, max_depth=5, min_child_samples=10,
             feature_fraction=0.7, bagging_fraction=0.85, bagging_freq=1, n_jobs=8, verbosity=-1,
             random_state=config.SEED)
CLF_B_ROUNDS = 300
REG_B = dict(objective="regression", learning_rate=0.04, num_leaves=31, max_depth=5, min_child_samples=10,
             feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, n_jobs=8, verbosity=-1,
             random_state=config.SEED)
REG_B_ROUNDS = 250


def load_b(which):
    """Temel tablodan NM'siz satırlar (eğitimde hedefi dolu olanlar). Yalnızca bu satırlar okunur."""
    path = config.DATA_DIR / f"base_v126_{which}_v1.parquet"
    flt = ds.field("NM_TAXI_DIFF_SEC").is_null()
    df = ds.dataset(path).to_table(filter=flt).to_pandas()
    if which == "train":
        df = df[df[TARGET_B].notna()].reset_index(drop=True)
    return df


def features_b(columns):
    """v126 listesi − TE − kimlikler − ikizler − NM'li satırlara özgü metinler; ADEP_mvt kategorik olarak eklenir."""
    return [c for c in F.feature_list(columns) if c not in DROP_B] + ["ADEP_mvt"]


def on_kosul_24s(df):
    """24 saat ön koşulu: kalkış tarifeden sonraki güne sarkmış (gecikme > günün saati) ve uçak tipi boş.
Eğitimde 68 satır (hepsi LIRF, 12'si 24 saat), testte 26 (hepsi LIRF)."""
    at = df["AIRCRAFT_TYPE_mvt"]
    empty = (at.isna() | at.astype(str).str.strip().isin(["", "nan", "None"])).to_numpy()
    return (df[DELAY].to_numpy(float) > df["MVT_MINOFDAY_UTC"].to_numpy(float) * 60) & empty


def regimes(df):
    t = df[TARGET_B].to_numpy(float)
    d = df[DELAY].to_numpy(float)
    r = np.zeros(len(df), dtype=int)
    r[(np.abs(t - d) <= 60) & (t >= 2000)] = 1
    r[t >= 80000] = 2
    return r


def _matrices(tr, ap, feats):
    X_tr, X_ap = tr[feats].copy(), ap[feats].copy()
    for c in feats:
        if c in CATS_B:
            s_tr, s_ap = FT._txt(X_tr[c]), FT._txt(X_ap[c])
            cats = sorted(s_tr.unique())
            X_tr[c], X_ap[c] = pd.Categorical(s_tr, categories=cats), pd.Categorical(s_ap, categories=cats)
        else:
            X_tr[c], X_ap[c] = FT._num(X_tr[c]), FT._num(X_ap[c])
    return X_tr, X_ap


# ----------------------------------------------------------------------------- v126'nın Model B'si (CV tabanı)
# PRC_4/src/train_model_b_production_grandmaster.py: gönderimlerdeki (v70 = … = v126) Model B'nin yöntemi.
V126B_NM = ["CALLSIGN_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "ADES_FILED", "LOBT_DIFF_SEC",
            "IOBT_DIFF_SEC", "EOBT_DIFF_SEC", "AOBT_DIFF_SEC", "ARVT3_DIFF_SEC", "AOBT_MINUS_EOBT", "NM_TAXI_DIFF_SEC",
            "DEST_VISIBILITY", "QUEUE_DELAY_EST", "NM_OFFSET_SEC", "NM_MISSING", "FLIGHT_TYPE_flt", "PLANNED_AIR_TIME",
            "AOBT_MINUS_IOBT"]                                                                                  # :42-48
V126B_EXC = {"MVT_ID_mvt", "TAXITIME_SEC_mvt", "VISIBILITY", "IS_LVP", "IS_LOW_VIS", "QUEUE_RATIO_TO_HOUR_MEAN",
             "SCHED_DIFF_SEC_HOUR_PCT", "delay", "ADEP_mvt"} | set(V126B_NM)                                 # :49-52
V126B_CATS = ["ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_CODE", "AIRLINE_LETTERS",
              "STAND_TYPE", "ROUTE_PAIR"]                                                                     # :56-59


def features_v126b(dep_columns):
    """v126 B özellikleri (:54): PRC_2 dep_train sütunlarından (config.DEP_TRAIN şeması) — 80 sütun."""
    return [c for c in dep_columns if c not in V126B_EXC and not c.startswith("DEST_") and "holiday" not in c.lower()]


def _v126b_frame(df, feats):
    X = df[feats].copy()
    for c in feats:
        X[c] = X[c].astype("category") if c in V126B_CATS else pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    return X                                                                                                  # :62-69


def fit_predict_v126b(tr, ap, feats):
    """v126'nın Model B yöntemi (:74-124): havalimanı bazlı; LIRF 3 rejimli karışım, diğerleri düz LGB + [60, 3.500].
    LIRF satırları için bileşenler de döner (p0, p1, p2, pred0, pred1, pred2; diğer satırlarda NaN)."""
    cats = [c for c in V126B_CATS if c in feats]
    X_tr, X_ap = _v126b_frame(tr, feats), _v126b_frame(ap, feats)
    out = np.full(len(ap), np.nan)
    comp = {k: np.full(len(ap), np.nan) for k in ("p0", "p1", "p2", "pred0", "pred1", "pred2")}
    for apt in ap["ADEP_mvt"].unique():
        ti, ai = (tr["ADEP_mvt"] == apt).to_numpy(), (ap["ADEP_mvt"] == apt).to_numpy()
        y = tr.loc[ti, TARGET_B].to_numpy(float)
        if apt == "LIRF":
            reg = np.zeros(len(y), dtype=int)
            reg[(y >= 2000) & (y < 40000)] = 1
            reg[y >= 40000] = 2
            clf = lgb.train(dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.03,
                                 num_leaves=31, max_depth=5, min_child_samples=6, feature_fraction=0.70, random_state=42,
                                 n_jobs=8, verbosity=-1),
                            lgb.Dataset(X_tr[ti], label=reg, categorical_feature=cats), num_boost_round=200)
            p = clf.predict(X_ap[ai])
            r0 = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.04, num_leaves=15, max_depth=4,
                                min_child_samples=6, verbosity=-1, n_jobs=8),
                           lgb.Dataset(X_tr[ti][reg == 0], label=y[reg == 0], categorical_feature=cats),
                           num_boost_round=100)
            d = ap.loc[ai, DELAY].to_numpy(float)
            parts = (r0.predict(X_ap[ai]), np.clip(d, 60.0, None), np.where(d > 40000, d, 87400.0))
            pred = p[:, 0] * parts[0] + p[:, 1] * parts[1] + p[:, 2] * parts[2]
            out[ai] = np.clip(pred, 60.0, None)
            for j in range(3):
                comp[f"p{j}"][ai], comp[f"pred{j}"][ai] = p[:, j], parts[j]
        else:
            m = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.04, num_leaves=31, max_depth=5,
                               min_child_samples=10, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1,
                               random_state=42, n_jobs=8, verbosity=-1),
                          lgb.Dataset(X_tr[ti], label=y, categorical_feature=cats), num_boost_round=250)
            out[ai] = np.clip(m.predict(X_ap[ai]), 60.0, 3500.0)
    return {"taksi": out, **comp}


# ----------------------------------------------------------------------------- tüm satırlarla (A + B) eğitim
# 29.09: NM'ye bağlı sütunlar dışındaki özellikler tüm eğitim satırlarında var. Ölçüm (29.09): yer
# tutucu Model A satırlarında 6.338 (%0,31), B'de 718 (%3,2); 24 saat kayması A'da 1, B'de 15. Bu yüzden p_yt ve μ tüm
# satırlardan, p_24 yalnızca B satırlarından öğrenilir. IS_B bayrağı özelliktir (taban oranları ~10 kat farklı).
CATS_TUM = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "STAND_TYPE", "AIRLINE_LETTERS"]
DROP_TUM = {"FLIGHT_CODE", "ROUTE_PAIR", "FLIGHT_NUMBER"}          # kimlik benzeri; tüm satırlı global modelde ezberlenir
REG_TUM = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, n_jobs=8, verbosity=-1, random_state=config.SEED)
REG_TUM_ROUNDS = 600
CLF_TUM = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, n_jobs=8, verbosity=-1, random_state=config.SEED)
CLF_TUM_ROUNDS = 400
CB_TUM_ITER = 3000


# Depo temel tablosunun NM'siz satırlarda bilgi taşıyan ek sütunları (29.09 ölçümü: 14 NM/CDM sütunu B'de hep 0,
# SCHED_MINUS_AOBT_v3 B'de −gecikmeye eşit; bunlar alınmaz)
EK_TUM = ["STAND_TO_RWY_DIST_M", "STAND_MIN_DIST_M", "STAND_DIST_RATIO", "DEP_COUNT_10", "DEP_COUNT_30", "DEP_COUNT_60",
          "DEP_RWY_COUNT_10", "DEP_RWY_COUNT_30", "RWY_PHYSICAL_LENGTH_FT", "RWY_HEADING_DEG_T", "RWY_DISPLACED_THRESH_FT"]


def features_tum(dep_columns, ek=False):
    """v126 B özellikleri − kimlik benzerleri + ADEP (kategorik) + IS_B; ek=True ise EK_TUM de."""
    return [c for c in features_v126b(dep_columns) if c not in DROP_TUM] + (EK_TUM if ek else []) + ["ADEP_mvt", "IS_B"]


def frame_tum(df, feats):
    X = df[feats].copy()
    for c in feats:
        X[c] = FT._txt(X[c]).astype("category") if c in CATS_TUM else pd.to_numeric(X[c], errors="coerce").astype(
            np.float32)
    return X


def fit_components_tum(tr, ap, feats, w_b=1.0, mu_huber=None, tohumlar=None, mu_cb=False, yt_cb=False, p24_cb=False,
                       cb_tohumlar=None):
    """tr: tüm satırlar (A + B, IS_B sütunuyla), ap: B satırları. Bileşenler:
    mu (tüm satırların normal rejimi), p24 ve yt_b (B satırlarıyla çok sınıflı; yt_b = P(yer tutucu | 24 saat değil)),
    yt_tum (tüm satırlarla ikili, 24 saat satırları hariç), d (GECIKME), d24 (24 saat senaryosundaki taksi).
    w_b: B satırlarının örnek ağırlığı (μ ve yt_tum eğitiminde); mu_huber: μ için Huber δ (None: MSE).
    tohumlar (07.10 gece): μ, rejim ve yt modelleri bu tohumlarla ayrı ayrı eğitilip tahminleri ortalanır
    (None: yalnız config.SEED, eski davranış).
    mu_cb (08.10): normal rejim için ayrıca CatBoost (GPU, kategorikler yerel) → "mu_cb" bileşeni (Model A'daki gibi
    LightGBM + CatBoost çeşitliliği; harman CV'de OOF bileşenleriyle seçilir).
    yt_cb (08.10): yer tutucu olasılığı için ayrıca CatBoost ikili sınıflandırıcı (GPU, Logloss) → "yt_tum_cb".
    p24_cb (08.10): rejim sınıflandırıcısı (yalnız B satırları, 3 sınıf) için ayrıca CatBoost (işlemci) → "p24_cb".
    cb_tohumlar (08.10): CatBoost μ'nün tohumları (ortalama; None: yalnız config.SEED)."""
    X_tr, X_ap = frame_tum(tr, feats), frame_tum(ap, feats)
    for c in CATS_TUM:                                   # uygulama kategorileri eğitimdekilerle aynı sırada
        X_ap[c] = pd.Categorical(X_ap[c].astype(object), categories=X_tr[c].cat.categories)
    y, r = tr[TARGET_B].to_numpy(float), regimes(tr)
    b = tr["IS_B"].to_numpy() == 1
    w = np.where(b, float(w_b), 1.0)
    keep = r != 2
    mu, p, yt = 0.0, 0.0, 0.0
    tohumlar = list(tohumlar or [config.SEED])
    for s in tohumlar:
        reg = dict(REG_TUM, random_state=s, **({"objective": "huber", "alpha": float(mu_huber)} if mu_huber else {}))
        mu = mu + lgb.train(reg, lgb.Dataset(X_tr[r == 0], label=y[r == 0], weight=w[r == 0]), REG_TUM_ROUNDS
                            ).predict(X_ap) / len(tohumlar)
        p = p + lgb.train(dict(CLF_B, random_state=s), lgb.Dataset(X_tr[b], label=r[b]), CLF_B_ROUNDS
                          ).predict(X_ap) / len(tohumlar)
        yt = yt + lgb.train(dict(CLF_TUM, random_state=s), lgb.Dataset(X_tr[keep], label=(r[keep] == 1).astype(float),
                                                                        weight=w[keep]), CLF_TUM_ROUNDS).predict(X_ap) / len(tohumlar)
    d = ap[DELAY].to_numpy(float)
    res24 = np.median(y[r == 2] - 86400) if (r == 2).any() else 1000.0
    out = {"mu": mu, "p24": p[:, 2], "yt_b": p[:, 1] / np.maximum(1e-9, 1 - p[:, 2]), "yt_tum": yt, "d": d,
           "d24": np.where(d > 40000, d, 86400 + res24)}
    if mu_cb or yt_cb or p24_cb:
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        def cbf(X):
            X = X.copy()
            for c in CATS_TUM:                       # eğitimde görülmeyen kategori NaN → "?" (CatBoost NaN kategori almaz)
                X[c] = X[c].astype(object).fillna("?").astype(str)
            return X
    if mu_cb:
        n = r == 0
        cbt = list(cb_tohumlar or [config.SEED])
        Xn, Xa = Pool(cbf(X_tr[n]), y[n], cat_features=CATS_TUM, weight=w[n]), Pool(cbf(X_ap), cat_features=CATS_TUM)
        out["mu_cb"] = np.mean([CatBoostRegressor(iterations=CB_TUM_ITER, learning_rate=0.08, depth=8, task_type="GPU",
                                                  random_seed=s, verbose=0).fit(Xn).predict(Xa) for s in cbt], axis=0)
    if yt_cb:
        m = CatBoostClassifier(iterations=CB_TUM_ITER, learning_rate=0.08, depth=8, task_type="GPU", random_seed=config.SEED,
                               verbose=0)
        m.fit(Pool(cbf(X_tr[keep]), (r[keep] == 1).astype(int), cat_features=CATS_TUM, weight=w[keep]))
        out["yt_tum_cb"] = m.predict_proba(Pool(cbf(X_ap), cat_features=CATS_TUM))[:, 1]
    if p24_cb:
        m = CatBoostClassifier(loss_function="MultiClass", iterations=1000, learning_rate=0.05, depth=6, random_seed=config.SEED,
                               verbose=0, thread_count=8)
        m.fit(Pool(cbf(X_tr[b]), r[b].astype(int), cat_features=CATS_TUM))
        pc = m.predict_proba(Pool(cbf(X_ap), cat_features=CATS_TUM))
        sinif = [str(int(float(c))) for c in m.classes_]
        assert "2" in sinif, f"p24_cb: eğitimde 24 saat sınıfı yok ({sinif})"
        out["p24_cb"] = pc[:, sinif.index("2")]
    return out


def combine_tum(c, adep, yt="yt_tum", cap_non_lirf=None):
    """Tahmin = p24·d24 + (1 − p24)·[yt·max(60, d) + (1 − yt)·mu]; LIRF dışı tavan isteğe bağlı; 60 sn alt sınır."""
    pred = c["p24"] * c["d24"] + (1 - c["p24"]) * (c[yt] * np.maximum(60.0, c["d"]) + (1 - c[yt]) * c["mu"])
    if cap_non_lirf is not None:
        pred = np.where(np.asarray(adep) == "LIRF", pred, np.minimum(pred, cap_non_lirf))
    return np.maximum(60.0, pred)


def fit_predict(tr, ap, feats, reg_scope="havalimani", mixture=True, cap_non_lirf=None):
    """Eğitim satırları tr ile ap satırlarının taksi tahmini (60 sn alt sınırlı) ve bileşenleri."""
    X_tr, X_ap = _matrices(tr, ap, feats)
    y = tr[TARGET_B].to_numpy(float)
    r = regimes(tr)
    out = {}
    normal = r == 0 if mixture else np.ones(len(tr), bool)
    mu = np.full(len(ap), np.nan)
    if reg_scope == "global":
        mu[:] = lgb.train(REG_B, lgb.Dataset(X_tr[normal], label=y[normal]), REG_B_ROUNDS).predict(X_ap)
    else:
        for apt in ap["ADEP_mvt"].unique():
            ti = (tr["ADEP_mvt"] == apt).to_numpy() & normal
            ai = (ap["ADEP_mvt"] == apt).to_numpy()
            mu[ai] = lgb.train(REG_B, lgb.Dataset(X_tr[ti], label=y[ti]), REG_B_ROUNDS).predict(X_ap[ai])
    out["mu_n"] = mu
    if mixture:
        p = lgb.train(CLF_B, lgb.Dataset(X_tr, label=r), CLF_B_ROUNDS).predict(X_ap)
        d = ap[DELAY].to_numpy(float)
        res24 = np.median(y[r == 2] - 86400) if (r == 2).any() else 1000.0
        d24 = np.where(d > 40000, d, 86400 + res24)
        out.update(p_yt=p[:, 1], p_24=p[:, 2])
        pred = p[:, 0] * mu + p[:, 1] * np.maximum(60.0, d) + p[:, 2] * d24
    else:
        pred = mu
    if cap_non_lirf is not None:
        pred = np.where(ap["ADEP_mvt"].eq("LIRF").to_numpy(), pred, np.minimum(pred, cap_non_lirf))
    out["taksi"] = np.maximum(60.0, pred)
    return out

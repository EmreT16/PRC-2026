"""Aday modeller: tek havalimanı için eğit ve tahmin et. Düzenekte (split) ve tam eğitimde (12 ay) aynı kod.

Aday tanımı (dict):
- ozellik: "v126" (134, metin → 0 ve NaN → 0 hatalarıyla) | "temiz" (C1–C3, TE'li) | "temiz_tesiz" (C4)
- modeller: ("lgb", "cb", "xgb") alt kümesi; tohumlar ve ayarlar v126'nınkiler (models_v126)
- cb_kat: CatBoost'a kategorikler de verilsin mi (v126'da yalnızca sayısal; max_ctr_complexity=1, A4)
- e1: kilitli satır karışımı: p·A + (1 − p)·harman_kilitsiz
- lgb_tohum: LightGBM tohumları (varsayılan v126'nın 42/123/777'si; E1 bileşeni düzenekte yalnızca 42 ile doğrulandı)
- lgb_huber: δ verilirse LightGBM Huber kaybıyla (objective=huber, alpha=δ; D2'de δ=1000 kabul)
- xgb_huber: δ verilirse XGBoost pseudo-Huber kaybıyla (objective=reg:pseudohubererror, huber_slope=δ)
- nan: True ise (yalnızca ozellik="v126") sayısallarda eksik değer 0 yerine NaN kalır (v126'nın NaN → 0 hatası kapalı)
- lgb_tur: LightGBM tur sayısı (varsayılan v126'nın 750'si)
- xgb_ek: XGBoost'a ek / değişen ayarlar sözlüğü (ör. n_estimators, subsample, colsample_bytree)
Harman: v126 ağırlıkları (0,45 CB / 0,40 LGB / 0,15 XGB), eksik bileşenlerde ağırlıklar yeniden ölçeklenir.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from . import config, features_temiz as FT, features_v126 as F, models_v126 as M

A_COL = "SCHED_MINUS_AOBT_v3"
LOCK_TOL = 60.0
CLF_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.75,
                  bagging_fraction=0.85, bagging_freq=1, n_jobs=8, verbosity=-1, random_state=config.SEED)
CLF_ROUNDS = 400


KUYRUK = ["KQ_TO_APT_PEN", "KQ_VARIS_PEN", "KQ_ONDE_APT", "KQ_ITME_20", "KQ_TO_PIST_PEN", "KQ_ONDE_PIST",
          "KQ_ONCEKI_PIST", "KQ_SONRAKI_PIST", "KQ_PIST_HIZ", "KQ_SERBEST"]  # scripts/kuyruk_ozellik.py (30.09)


def features(spec, columns):
    if spec["ozellik"] == "v126":
        f = F.feature_list(columns)
    else:
        f = FT.feature_list(columns, te=spec["ozellik"] == "temiz")
    if spec.get("kuyruk"):
        f = f + [c for c in KUYRUK if c in columns and c not in f]
    return f


PLAN = ["PL_LOBT", "PL_IOBT", "PL_EOBT"]  # LOBT / IOBT / EOBT_1 − AOBT_3 (sn); scripts/plan_ozellik.py (30.09)


def kuyruk_ekle(spec, *dfs):
    """kuyruk=True adaylarda NM off-block ↔ kalkış penceresi özelliklerini (data/kuyruk_oz.parquet) ekler;
    e1p=True adaylarda planlı saat farklarını (data/plan_oz.parquet) da."""
    if spec.get("e1p") and not all(c in dfs[0].columns for c in PLAN):
        pl = pd.read_parquet(config.DATA_DIR / "plan_oz.parquet")
        dfs = tuple(d.merge(pl, on="MVT_ID_mvt", how="left", validate="one_to_one") for d in dfs)
    if not spec.get("kuyruk") or all(c in dfs[0].columns for c in KUYRUK):
        return dfs
    kq = pd.read_parquet(config.DATA_DIR / "kuyruk_oz.parquet", columns=["MVT_ID_mvt"] + KUYRUK)
    for c in KUYRUK:
        kq[c] = kq[c].astype(np.float32)
    return tuple(d.merge(kq, on="MVT_ID_mvt", how="left", validate="one_to_one") for d in dfs)


def _matrices(spec, tr, ap, feats):
    """(LightGBM matrisi eğitim/uygulama, sayısal dizi eğitim/uygulama, kategorik sütunlar)."""
    if spec["ozellik"] == "v126":
        kn = spec.get("nan", False)
        X_tr, X_ap = F.matrix(tr, feats, keep_nan=kn), F.matrix(ap, feats, keep_nan=kn)
        (n_tr, num), (n_ap, _) = F.numeric_array(X_tr, feats, kn), F.numeric_array(X_ap, feats, kn)
        return X_tr, X_ap, n_tr, n_ap, num, F.CAT_V126
    X_tr, X_ap = FT.matrices(tr, ap, feats)
    cats = [c for c in feats if c in FT.CATS]
    num = [c for c in feats if c not in FT.CATS]
    return X_tr, X_ap, X_tr[num].to_numpy(np.float32), X_ap[num].to_numpy(np.float32), num, cats


def _cb_frame(n, num, X, cats):
    return pd.concat([pd.DataFrame(n, columns=num), X[cats].astype(str).reset_index(drop=True)], axis=1)


def _regress(spec, X_tr, X_ap, n_tr, n_ap, num, cats, y, w=None):
    out = {}
    if "lgb" in spec["modeller"]:
        seeds = spec.get("lgb_tohum", M.SEEDS["lgb"])
        params = dict(M.LGB_PARAMS)
        if spec.get("lgb_huber"):
            params.update(objective="huber", alpha=float(spec["lgb_huber"]))
        params.update(spec.get("lgb_ek", {}))  # 09.10: LTFM taraması (scripts/ltfm_tarama.py); varsayılan boş
        out["lgb"] = np.mean([lgb.train({**params, "random_state": s}, lgb.Dataset(X_tr, label=y, weight=w),
                                        num_boost_round=spec.get("lgb_tur", M.LGB_ROUNDS)).predict(X_ap)
                              for s in seeds], axis=0)
    if "cb" in spec["modeller"]:
        params = dict(iterations=650, learning_rate=0.045, depth=8, task_type="GPU", verbose=0)
        if spec.get("cb_kat"):
            f_tr, f_ap = _cb_frame(n_tr, num, X_tr, cats), _cb_frame(n_ap, num, X_ap, cats)
            out["cb"] = np.mean([CatBoostRegressor(**params, random_seed=s, max_ctr_complexity=1)
                                 .fit(Pool(f_tr, y, cat_features=cats)).predict(Pool(f_ap, cat_features=cats))
                                 for s in M.SEEDS["cb"]], axis=0)
        else:
            out["cb"] = np.mean([CatBoostRegressor(**params, random_seed=s).fit(n_tr, y).predict(n_ap)
                                 for s in M.SEEDS["cb"]], axis=0)
    if "xgb" in spec["modeller"]:
        xp = dict(n_estimators=500, learning_rate=0.04, max_depth=7, tree_method="hist", device="cuda", verbosity=0)
        if spec.get("xgb_huber"):
            xp.update(objective="reg:pseudohubererror", huber_slope=float(spec["xgb_huber"]))
        xp.update(spec.get("xgb_ek", {}))
        out["xgb"] = np.mean([xgb.XGBRegressor(**xp, random_state=s).fit(n_tr, y).predict(n_ap)
                              for s in M.SEEDS["xgb"]], axis=0)
    w = {k: M.WEIGHTS[k] for k in out}
    out["harman"] = sum(w[k] * out[k] for k in out) / sum(w.values())
    return out


def mevsim_agirligi(spec, tr, ap):
    """mevsim=w (01.10 gece): tahmin edilen aylara (ap) dairesel uzaklığı ≤ 1 olan eğitim aylarına w ağırlığı (yalnız LGB).
    Sınamada (ör. split_1: Oca/Tem) komşu aylar (Ara, Şub, Haz, Ağu); tam eğitimde (test Oca/Tem) aynı aylar ve komşuları."""
    if not spec.get("mevsim"):
        return None
    hedef = set(ap["MVT_MONTH"].astype(int).unique().tolist())
    m = tr["MVT_MONTH"].astype(int).to_numpy()
    uz = np.min([np.minimum(np.abs(m - t), 12 - np.abs(m - t)) for t in hedef], axis=0)
    return np.where(uz <= 1, float(spec["mevsim"]), 1.0)


def fit_predict_airport(spec, tr, ap, feats):
    """Tek havalimanı: eğitim satırları tr (hedefli), uygulama satırları ap. Ofset tahminleri sözlüğü."""
    y = tr[config.TARGET].to_numpy(float)
    X_tr, X_ap, n_tr, n_ap, num, cats = _matrices(spec, tr, ap, feats)
    if spec.get("hedef_taksi"):  # 30.09: taksiyi doğrudan tahmin et (NM taksi özellik olarak), sonra ofsete çevir
        nm_tr, nm_ap = tr["NM_TAXI_DIFF_SEC"].to_numpy(float), ap["NM_TAXI_DIFF_SEC"].to_numpy(float)
        X_tr, X_ap = X_tr.assign(NM_TAKSI=nm_tr), X_ap.assign(NM_TAKSI=nm_ap)
        n_tr, n_ap = np.column_stack([n_tr, nm_tr]).astype(np.float32), np.column_stack([n_ap, nm_ap]).astype(np.float32)
        out = _regress(spec, X_tr, X_ap, n_tr, n_ap, num + ["NM_TAKSI"], cats, nm_tr - y)
        return {k: nm_ap - v for k, v in out.items()}
    if spec.get("e1p"):
        return _e1p(spec, tr, ap, X_tr, X_ap, n_tr, n_ap, num, cats, y)
    if not spec.get("e1"):
        return _regress(spec, X_tr, X_ap, n_tr, n_ap, num, cats, y, mevsim_agirligi(spec, tr, ap))
    lock = np.abs(y - tr[A_COL].to_numpy(float)) <= LOCK_TOL
    p = lgb.train(CLF_PARAMS, lgb.Dataset(X_tr, label=lock.astype(float)), num_boost_round=CLF_ROUNDS).predict(X_ap)
    keep = ~lock
    out = _regress(spec, X_tr[keep], X_ap, n_tr[keep], n_ap, num, cats, y[keep])
    out["mu_kilitsiz"] = out["harman"]
    out["p_kilit"] = p
    out["harman"] = p * ap[A_COL].to_numpy(float) + (1 - p) * out["mu_kilitsiz"]
    return out


def _e1p(spec, tr, ap, X_tr, X_ap, n_tr, n_ap, num, cats, y):
    """E1'in genişletilmiş hâli (30.09): kilitli sınıfları tarife (SCHED) + planlı saat yankıları (LOBT, IOBT, EOBT;
    tarifeye eşit olmayanlar). Çok sınıflı sınıflandırıcı; tahmin = Σ p_k · yankı_ofseti_k + p_0 · μ_kilitsiz.
    Eğitimde 50'den az örneği olan sınıf kullanılmaz (normal sayılır)."""
    degerler = [tr[A_COL].to_numpy(float)] + [tr[c].to_numpy(float) for c in PLAN]
    ap_deger = [ap[A_COL].to_numpy(float)] + [ap[c].to_numpy(float) for c in PLAN]
    sinif = np.zeros(len(y), dtype=int)
    for k, v in enumerate(degerler, start=1):
        sinif[(sinif == 0) & np.isfinite(v) & (np.abs(y - v) <= LOCK_TOL)] = k
    kullan = [k for k in range(1, len(degerler) + 1) if (sinif == k).sum() >= 50]
    harita = {k: i + 1 for i, k in enumerate(kullan)}
    lab = np.array([harita.get(s, 0) for s in sinif])
    ek_tr = pd.DataFrame({f"d_{c}": tr[A_COL].to_numpy(float) - tr[c].to_numpy(float) for c in PLAN}, index=X_tr.index)
    ek_ap = pd.DataFrame({f"d_{c}": ap[A_COL].to_numpy(float) - ap[c].to_numpy(float) for c in PLAN}, index=X_ap.index)
    Xc_tr = pd.concat([X_tr, tr[PLAN].set_axis(X_tr.index).astype(float), ek_tr], axis=1)
    Xc_ap = pd.concat([X_ap, ap[PLAN].set_axis(X_ap.index).astype(float), ek_ap], axis=1)
    n_sinif = len(kullan) + 1
    if n_sinif == 1:
        P = np.ones((len(ap), 1))
    elif n_sinif == 2:
        p1 = lgb.train(CLF_PARAMS, lgb.Dataset(Xc_tr, label=(lab == 1).astype(float)), num_boost_round=CLF_ROUNDS).predict(Xc_ap)
        P = np.column_stack([1 - p1, p1])
    else:
        prm = dict(CLF_PARAMS, objective="multiclass", num_class=n_sinif)
        P = lgb.train(prm, lgb.Dataset(Xc_tr, label=lab), num_boost_round=CLF_ROUNDS).predict(Xc_ap)
    keep = lab == 0
    out = _regress(spec, X_tr[keep], X_ap, n_tr[keep], n_ap, num, cats, y[keep])
    mu = out["harman"]
    vals = [mu] + [np.where(np.isfinite(ap_deger[k - 1]), ap_deger[k - 1], mu) for k in kullan]
    out["mu_kilitsiz"] = mu
    out["p_kilit"] = P[:, harita[1]] if 1 in harita else np.zeros(len(ap))
    out["p_plan"] = P[:, 1:].sum(axis=1) - out["p_kilit"]
    out["harman"] = sum(P[:, i] * vals[i] for i in range(n_sinif))
    return out


# Aday kayıtları. "v126" düzenekte B1'in aynısıdır; tam eğitimde v126 YENİ_200'ün 12 aylık hâli.
ADAYLAR = {
    "v126": dict(ozellik="v126", modeller=("lgb", "cb", "xgb")),
    "v126_e1": dict(ozellik="v126", modeller=("lgb", "cb", "xgb"), e1=True),
    "temiz": dict(ozellik="temiz", modeller=("lgb", "cb", "xgb")),
    "temiz_e1": dict(ozellik="temiz", modeller=("lgb", "cb", "xgb"), e1=True),
    "temiz_cbkat": dict(ozellik="temiz", modeller=("lgb", "cb", "xgb"), cb_kat=True),
    "temiz_cbkat_e1": dict(ozellik="temiz", modeller=("lgb", "cb", "xgb"), cb_kat=True, e1=True),
    # G1 beş bileşenli harmanın 12 aylık parçaları (CB'nin ağırlığı 0; global CB ayrı: d_global_cb.py)
    "tam_lgb_xgb": dict(ozellik="v126", modeller=("lgb", "xgb")),
    "tam_e1_lgb42": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_tohum=(42,)),
    "tam_lgb_huber1000": dict(ozellik="v126", modeller=("lgb",), lgb_tohum=(42,), lgb_huber=1000),
    # Huber'in diğer bileşenlere yayılması (düzenekte doğrulanacak; 29.09)
    "v_xgb_huber1000": dict(ozellik="v126", modeller=("xgb",), xgb_huber=1000),
    "v_e1_huber1000": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_tohum=(42,), lgb_huber=1000),
    "v_lgb_huber1000_3t": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000),
    "v_e1_huber1000_3t": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000),
    # 30.09: kuyruk / pencere özellikleri (scripts/kuyruk_ozellik.py) eklenmiş bileşenler
    "v_lgb_huber1000_3t_kq": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True),
    "v_e1_huber1000_3t_kq": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000, kuyruk=True),
    "v_xgb_kq": dict(ozellik="v126", modeller=("xgb",), kuyruk=True),
    "v_lgb_huber1000_3t_kqg": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True),
    "v_e1_huber1000_3t_kqg": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000, kuyruk=True),
    "v_xgb_kqg": dict(ozellik="v126", modeller=("xgb",), kuyruk=True),
    "v_lgb_huber1000_3t_kqgp": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True),  # 01.10: + pist bazlı gecikme endeksi (PRC_GECPIST=1)
    "v_e1_huber1000_3t_kqgp": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000, kuyruk=True),
    "v_xgb_kqgp": dict(ozellik="v126", modeller=("xgb",), kuyruk=True),
    "v_xgb_kqgp_1500": dict(ozellik="v126", modeller=("xgb",), kuyruk=True, xgb_ek=dict(n_estimators=1500)),  # 01.10: kapasite
    # 02.10 akşam: + yüzey/kar özellikleri (PRC_EKOZ=1, kutuk/ekoz_secim.json)
    "v_lgb_huber1000_3t_kqgpy": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True),
    "v_e1_huber1000_3t_kqgpy": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000, kuyruk=True),
    "v_xgb_kqgpy_1500": dict(ozellik="v126", modeller=("xgb",), kuyruk=True, xgb_ek=dict(n_estimators=1500)),
    # 05.10: test tarafında ek LightGBM tohumları (1, 2, 3); _3t'li dosyayla ortalaması 6 tohum (scripts/zincir_lgbtohum.sh)
    "v_lgb_huber1000_3t_kqgpy_b": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True, lgb_tohum=(1, 2, 3)),
    "v_e1_huber1000_3t_kqgpy_b": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000, kuyruk=True,
                                      lgb_tohum=(1, 2, 3)),
    # 05.10: pist konfigürasyonlu (pist_oz) ek tohumlar — scripts/zincir_pist_tohum.sh (ekoz_secim'e pist_oz eklenir)
    "v_lgb_huber1000_3t_kqgpy_pb": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True, lgb_tohum=(1, 2, 3)),
    "v_e1_huber1000_3t_kqgpy_pb": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000, kuyruk=True,
                                       lgb_tohum=(1, 2, 3)),
    # 06.10: final yapısı (pist dahil, PRC_EKOZ_DOSYA=ekoz_secim_final.json) MVT_DAY'siz (PRC_GUNSUZ=1);
    # 6 tohum tek koşuda (42, 123, 777 + 1, 2, 3; iki 3'lü koşunun ortalamasıyla aynı)
    "v_lgb_huber1000_6t_kqgpy_gunsuz": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True,
                                            lgb_tohum=(42, 123, 777, 1, 2, 3)),
    "v_e1_huber1000_6t_kqgpy_gunsuz": dict(ozellik="v126", modeller=("lgb",), e1=True, lgb_huber=1000, kuyruk=True,
                                           lgb_tohum=(42, 123, 777, 1, 2, 3)),
    "v_xgb_kqgpy_1500_gunsuz": dict(ozellik="v126", modeller=("xgb",), kuyruk=True, xgb_ek=dict(n_estimators=1500)),
    "v_lgb_huber1000_3t_kqgp_1500": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True, lgb_tur=1500),
    "v_lgb_huber1000_3t_kqgp_mev2": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True, mevsim=2.0),  # 01.10: mevsim ağırlığı
    "v_lgb_taksi_kq": dict(ozellik="v126", modeller=("lgb",), lgb_huber=1000, kuyruk=True, hedef_taksi=True),
    "v_e1p_huber1000_3t_kq": dict(ozellik="v126", modeller=("lgb",), e1p=True, lgb_huber=1000, kuyruk=True),
    "v_lgb_huber1000_nan": dict(ozellik="v126", modeller=("lgb",), lgb_tohum=(42,), lgb_huber=1000, nan=True),
    "v_xgb_nan": dict(ozellik="v126", modeller=("xgb",), nan=True),
    # Tur sayısı / düzenlileştirme (29.09 akşam)
    "v_xgb_1500": dict(ozellik="v126", modeller=("xgb",), xgb_ek=dict(n_estimators=1500)),
    "v_xgb_1500_alt": dict(ozellik="v126", modeller=("xgb",),
                           xgb_ek=dict(n_estimators=1500, subsample=0.85, colsample_bytree=0.75, min_child_weight=5)),
    "v_lgb_huber1000_1500": dict(ozellik="v126", modeller=("lgb",), lgb_tohum=(42,), lgb_huber=1000, lgb_tur=1500),
}

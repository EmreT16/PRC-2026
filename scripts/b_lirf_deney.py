"""09.10 öğleden sonra: LIRF Model B rejim sınıflandırıcısı değişkenleri.

Neden: LIRF NM'siz satırları Oca / Tem testinde 383 satır ama CV RMSE ≈ 3.000 → toplam SSE'nin kabaca %20'si. Hata çoğunlukla
rejim karışmasından: gecikmeli satırların ~yarısında kayıtlı taksi = gecikme (BLOCK tarifeden kopya; ADS-B ayırmıyor, fiziksel
taksi iki rejimde aynı), öbür yarısında normal taksi. Tahmin p0 · r0 + p1 · gecikme + p2 · (24 sa) → p'nin kalitesi belirleyici.
Sınıflandırıcı ~1.500 satırda 80 özellik + yüksek kardinaliteli kategoriler (FLIGHT_CODE, ROUTE_PAIR, STAND) ile eğitiliyor.
Değişkenler (hepsi aynı çeyrek CV'si, ön koşul dışı satırlar; her değişken 5 tohum ortalaması → tohum gürültüsü ayrılır):
  taban   üretim (LightGBM, v126b özellikleri + oran hy / park)
  kks     yüksek kardinaliteli kategoriler (FLIGHT_CODE, ROUTE_PAIR, STAND_mvt) çıkarılmış
  duz     daha güçlü düzenlileştirme (min_child_samples 20, lambda_l2 5, cat_smooth 20, min_data_per_group 20)
  cb      CatBoost (derinlik 5, 600 yineleme, lr 0,03; kategoriler CatBoost'un sıralı kodlamasıyla)
  lgbcb   taban + cb olasılık ortalaması
  boran   taban + yalnız NM'siz LIRF satırlarından havayolu kopya oranı (eğitim katında ay dışı)
  agir / agir2 / kksagir / kksagir2  satır ağırlığı |gecikme − 1.000| (¹ / ²); kks ile birlikte
Kabul: ön koşul dışı RMSE iyi ve 4/4 çeyrek (en az 3/4 ve toplam belirgin) iyi.
Kullanım: python scripts/b_lirf_deney.py [degiskenler=taban,kks,...] [tohum=5]
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config, model_b as MB, validation as V  # noqa: E402
import b_lirf_te as L  # noqa: E402

YUKSEK = ["FLIGHT_CODE", "ROUTE_PAIR", "STAND_mvt"]
LGB_P = dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.03, num_leaves=31, max_depth=5,
             min_child_samples=6, feature_fraction=0.70, n_jobs=8, verbosity=-1)


def b_oran(tr, hedef, oof):
    """Yalnız NM'siz LIRF eğitim satırlarından havayolu kopya oranı (rejim 1), Bayes yumuşatmalı (öncül 10)."""
    y = tr[MB.TARGET_B].to_numpy(float)
    k = pd.Series(((y >= 2000) & (y < 40000)).astype(float), index=tr.index)
    hy_tr, hy_h = L.anahtarlar(tr["FLIGHT_mvt"])["hy"], L.anahtarlar(hedef["FLIGHT_mvt"])["hy"]
    out = np.zeros(len(hedef))
    aylar = np.unique(hedef["MVT_MONTH"]) if oof else [None]
    for m in aylar:
        s = (tr["MVT_MONTH"] != m).to_numpy() if m is not None else np.ones(len(tr), bool)
        i = (hedef["MVT_MONTH"] == m).to_numpy() if m is not None else np.ones(len(hedef), bool)
        p0 = k[s].mean()
        g = k[s].groupby(hy_tr[s]).agg(["sum", "size"])
        o = (g["sum"] + 10 * p0) / (g["size"] + 10)
        out[i] = hy_h[i].map(o).fillna(p0).to_numpy(float)
    return out


def olasilik(degisken, X_tr, reg, X_ap, cats, tohum, w=None):
    if degisken in ("taban", "kks", "boran") or degisken.startswith(("agir", "kksagir")):
        return np.mean([lgb.train(dict(LGB_P, random_state=42 + s), lgb.Dataset(X_tr, label=reg, weight=w,
                                                                               categorical_feature=cats),
                                  200).predict(X_ap) for s in range(tohum)], axis=0)
    if degisken == "duz":
        P = dict(LGB_P, min_child_samples=20, lambda_l2=5.0, cat_smooth=20, min_data_per_group=20)
        return np.mean([lgb.train(dict(P, random_state=42 + s), lgb.Dataset(X_tr, label=reg, categorical_feature=cats),
                                  200).predict(X_ap) for s in range(tohum)], axis=0)
    if degisken == "cb":
        from catboost import CatBoostClassifier
        A, B = X_tr.copy(), X_ap.copy()
        for c in cats:
            A[c], B[c] = A[c].astype(object).fillna("?").astype(str), B[c].astype(object).fillna("?").astype(str)
        ps = []
        for s in range(tohum):
            m = CatBoostClassifier(loss_function="MultiClass", depth=5, iterations=600, learning_rate=0.03, l2_leaf_reg=5,
                                   random_seed=42 + s, verbose=0, thread_count=8, cat_features=cats)
            m.fit(A, reg)
            p = m.predict_proba(B)
            if p.shape[1] < 3:  # eğitim katında 2. rejim yoksa
                p = np.column_stack([p, np.zeros((len(p), 3 - p.shape[1]))])
            ps.append(p)
        return np.mean(ps, axis=0)
    if degisken == "lgbcb":
        return 0.5 * olasilik("taban", X_tr, reg, X_ap, cats, tohum) + 0.5 * olasilik("cb", X_tr, reg, X_ap, cats, tohum)
    raise SystemExit(f"bilinmeyen değişken {degisken}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    degiskenler = opt.get("degiskenler", "taban,kks,duz,cb,lgbcb,boran").split(",")
    tohum = int(opt.get("tohum", 5))
    ref = L.lirf_ham()
    df = L.b_satirlari("train")
    feats0 = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    y = df[MB.TARGET_B].to_numpy(float)
    pre = MB.on_kosul_24s(df)
    d = df[MB.DELAY].to_numpy(float)
    tahmin = {k: np.full(len(df), np.nan) for k in degiskenler}
    ll = {k: [] for k in degiskenler}
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr, va = df[~va_m].reset_index(drop=True), df[va_m].reset_index(drop=True)
        ref_q = ref[~ref["ay"].isin(V.QUARTERS[q])]
        ek_tr, ek_va = L.te_ozellik(tr, ref_q, oof=True), L.te_ozellik(va, ref_q, oof=False)
        ytr = tr[MB.TARGET_B].to_numpy(float)
        reg = np.where(ytr >= 40000, 2, np.where(ytr >= 2000, 1, 0))
        # rejim 0 regresyonu ve diğer parçalar: b_lirf_te.fit_lirf ile aynı
        cats0 = [c for c in MB.V126B_CATS if c in feats0]
        Xr_tr, Xr_va = MB._v126b_frame(tr, feats0), MB._v126b_frame(va, feats0)
        r0 = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.04, num_leaves=15, max_depth=4,
                            min_child_samples=6, verbosity=-1, n_jobs=8),
                       lgb.Dataset(Xr_tr[reg == 0], label=ytr[reg == 0], categorical_feature=cats0), num_boost_round=100)
        dv = va[MB.DELAY].to_numpy(float)
        parts = (r0.predict(Xr_va), np.clip(dv, 60.0, None), np.where(dv > 40000, dv, 87400.0))
        yv = va[MB.TARGET_B].to_numpy(float)
        regv = np.where(yv >= 40000, 2, np.where(yv >= 2000, 1, 0))
        for ad in degiskenler:
            feats = [c for c in feats0 if not (ad.startswith("kks") and c in YUKSEK)]
            cats = [c for c in MB.V126B_CATS if c in feats]
            X_tr = pd.concat([MB._v126b_frame(tr, feats).reset_index(drop=True), ek_tr.reset_index(drop=True)], axis=1)
            X_va = pd.concat([MB._v126b_frame(va, feats).reset_index(drop=True), ek_va.reset_index(drop=True)], axis=1)
            if ad == "boran":
                X_tr["YT_ORAN_HY_B"] = b_oran(tr, tr, oof=True)
                X_va["YT_ORAN_HY_B"] = b_oran(tr, va, oof=False)
            # 09.10: agir* — sınıflandırıcı satır ağırlığı, yanlış rejimin karesel bedeliyle orantılı (gecikme − 1.000 sn):
            # agir = |·|, agir2 = (·)², ortalaması 1'e ölçekli; logloss'u RMSE'nin baskın satırlarına hizalar.
            w = None
            if "agir" in ad:
                dt = tr[MB.DELAY].to_numpy(float)
                w = np.abs(dt - 1000.0) ** (2 if ad.endswith("2") else 1) + 1.0
                w = w / w.mean()
            p = olasilik(ad, X_tr, reg, X_va, cats, tohum, w)
            tahmin[ad][va_m] = np.clip(p[:, 0] * parts[0] + p[:, 1] * parts[1] + p[:, 2] * parts[2], 60.0, None)
            ll[ad].append(-np.mean(np.log(np.clip(p[np.arange(len(regv)), regv], 1e-6, 1))))
        print(f"{q} tamam", flush=True)
    m = ~pre
    print(f"\nLIRF ön koşul dışı {m.sum()} satır, {tohum} tohum ortalaması")
    b = tahmin[degiskenler[0]]
    for ad in degiskenler:
        p = tahmin[ad]
        cey = []
        for qn, ms in V.QUARTERS.items():
            mm = df["MVT_MONTH"].isin(ms).to_numpy() & m
            cey.append(np.sqrt(np.mean((p[mm] - y[mm]) ** 2)))
        iyi = sum(np.sqrt(np.mean((p[df["MVT_MONTH"].isin(ms).to_numpy() & m] - y[df["MVT_MONTH"].isin(ms).to_numpy() & m]) ** 2))
                  < np.sqrt(np.mean((b[df["MVT_MONTH"].isin(ms).to_numpy() & m] - y[df["MVT_MONTH"].isin(ms).to_numpy() & m]) ** 2))
                  for ms in V.QUARTERS.values())
        print(f"{ad:<7} RMSE {np.sqrt(np.mean((p[m] - y[m]) ** 2)):7.1f} | logloss {np.mean(ll[ad]):.4f} | çeyrekler "
              + " ".join(f"{c:7.1f}" for c in cey) + f" | tabana göre iyi çeyrek {iyi}/4")
    pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], **{f"p_{k}": v for k, v in tahmin.items()}}).to_parquet(
        config.DATA_DIR / "preds" / "B_lirf_deney_cv.parquet", index=False)


if __name__ == "__main__":
    main()

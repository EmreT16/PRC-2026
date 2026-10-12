"""İterasyon eğrisi (04.10 gece). Standart yöntem: ay dışı bölmede yüksek tur sınırıyla
bir kez eğit, her k turda doğrulama hatasını ölç; en iyi tur = eğrinin en düşük noktası. Yalnız 2025 eğitim verisi, LB yok.
Tek tohum (42), havalimanı başına model (üretimdeki gibi, src/aday.py fit_predict_airport). Hata gönderilen taksi üzerinden
(60 sn alt sınırlı, validation.floor_offset).

Kullanım: python scripts/iter_egrisi.py <lgb|xgb> split_1 split_2 ...
  lgb: lgbh3 (LightGBM Huber 1000, tüm satırlar) + E1 (kilit sınıflandırıcısı × kilitsiz satırlarda LightGBM) aynı süreçte
  xgb: XGBoost (ekran kartı)
Çıktı: kutuk/iter_egrisi_<bilesen>.parquet (bilesen, split, apt, ay, kc, k, sse, n); kc = E1 sınıflandırıcı turu."""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import aday as AD, config, features_v126 as F, models_v126 as M, validation as V  # noqa: E402

LGB_K = list(range(100, 1601, 50))
CLF_K = [100, 200, 300, 400, 600, 800, 1000]
XGB_K = list(range(250, 4001, 250))
SPEC = {"lgb": AD.ADAYLAR["v_e1_huber1000_3t_kqgpy"], "xgb": AD.ADAYLAR["v_xgb_kqgpy_1500"]}
LGBP = dict(M.LGB_PARAMS, objective="huber", alpha=1000.0, random_state=42, n_jobs=12)


def sse(pred, yv, nm, mon):
    e2 = (V.floor_offset(pred, nm) - yv) ** 2
    return {m: (float(e2[mon == m].sum()), int((mon == m).sum())) for m in np.unique(mon)}


def kaydet(satir, bil, split, apt, kc, k, d):
    for m, (s, n) in d.items():
        satir.append((bil, split, apt, int(m), kc, k, s, n))


def bolme(bil, split, cikti):
    t0 = time.time()
    train, val, vm = F.prepare_split(split)
    spec = SPEC[bil]
    train, val = AD.kuyruk_ekle(spec, train, val)
    feats = AD.features(spec, train.columns)
    satir = []
    for apt in sorted(val["ADEP_mvt"].unique()):
        ta = time.time()
        tr, ap = train[train["ADEP_mvt"] == apt], val[val["ADEP_mvt"] == apt]
        y = tr[config.TARGET].to_numpy(float)
        yv, nm = ap[config.TARGET].to_numpy(float), ap["NM_TAXI_DIFF_SEC"].to_numpy(float)
        mon = ap["MVT_MONTH"].to_numpy()
        X_tr, X_ap, n_tr, n_ap, num, cats = AD._matrices(spec, tr, ap, feats)
        if bil == "xgb":
            xp = dict(n_estimators=max(XGB_K), learning_rate=0.04, max_depth=7, tree_method="hist", device="cuda",
                      verbosity=0, random_state=42)
            m = xgb.XGBRegressor(**xp).fit(n_tr, y)
            for k in XGB_K:
                kaydet(satir, "xgb", split, apt, 0, k, sse(m.predict(n_ap, iteration_range=(0, k)), yv, nm, mon))
        else:
            b = lgb.train(LGBP, lgb.Dataset(X_tr, label=y), num_boost_round=max(LGB_K))
            for k in LGB_K:
                kaydet(satir, "lgbh3", split, apt, 0, k, sse(b.predict(X_ap, num_iteration=k), yv, nm, mon))
            A = tr[AD.A_COL].to_numpy(float)
            lock = np.abs(y - A) <= AD.LOCK_TOL
            c = lgb.train(dict(AD.CLF_PARAMS, n_jobs=12), lgb.Dataset(X_tr, label=lock.astype(float)),
                          num_boost_round=max(CLF_K))
            P = {kc: c.predict(X_ap, num_iteration=kc) for kc in CLF_K}
            r = lgb.train(LGBP, lgb.Dataset(X_tr[~lock], label=y[~lock]), num_boost_round=max(LGB_K))
            A_ap = ap[AD.A_COL].to_numpy(float)
            for k in LGB_K:
                mu = r.predict(X_ap, num_iteration=k)
                for kc in CLF_K:
                    kaydet(satir, "e1", split, apt, kc, k, sse(P[kc] * A_ap + (1 - P[kc]) * mu, yv, nm, mon))
        print(f"  {split} {apt}: {time.time() - ta:.0f} sn", flush=True)
    d = pd.DataFrame(satir, columns=["bilesen", "split", "apt", "ay", "kc", "k", "sse", "n"])
    if cikti.exists():
        d = pd.concat([pd.read_parquet(cikti).query("split != @split"), d], ignore_index=True)
    d.to_parquet(cikti, index=False)
    print(f"[{split}] süre {time.time() - t0:.0f} sn", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    bil = sys.argv[1]
    for s in sys.argv[2:]:
        bolme(bil, s, config.ROOT / "kutuk" / f"iter_egrisi_{bil}.parquet")

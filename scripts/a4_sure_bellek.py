"""A4 — v126 tri-model ayarlarının süre ve GPU belleği ölçümü (tek havalimanı, split_1 eğitim ayları).

v126 ayarları (train_and_build_v126_leaves200.py:228-305):
- LightGBM: 200 yaprak, lr 0,035, 750 tur; n_jobs 8 (v126) ve 16.
- CatBoost GPU: depth 8, lr 0,045, 650 iterasyon. v126 yalnızca sayısal sütun veriyordu; ayrıca 5 kategorikle
  max_ctr_complexity varsayılan ve 1 denenir ("1 olmazsa bellek taşar" iddiasının sınaması).
- XGBoost CUDA: depth 7, lr 0,04, 500 ağaç.
Hatalar yutulmaz: her model açıkça başarılı/başarısız raporlanır (v126:293-308'deki sessiz yedeklemenin tersine).
Doğrulama RMSE'si yalnızca bilgi amaçlıdır (model kararı değil).
"""
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, validation as V  # noqa: E402

LGB_V126 = dict(objective="regression", metric="rmse", learning_rate=0.035, num_leaves=200, max_depth=-1,
                min_child_samples=25, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, verbosity=-1)


class GpuPeak:
    """nvidia-smi ile 0,5 sn aralıkla GPU bellek kullanımının tepe değerini (MiB) izler."""

    def __enter__(self):
        self.peak, self._stop = 0, threading.Event()
        self.base = self._used()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _used(self):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[0])

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(0.5)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def run(name, fn, y_va, months_va, results):
    """Modeli eğitir; süre, GPU tepe belleği ve doğrulama RMSE'sini kaydeder. Hata yutulmaz, raporlanır."""
    t = time.time()
    try:
        with GpuPeak() as g:
            pred = fn()
        rmse = float(np.sqrt(np.mean((pred - y_va) ** 2)))
        per_m = {int(m): round(float(np.sqrt(np.mean((pred[months_va == m] - y_va[months_va == m]) ** 2))), 2)
                 for m in (1, 7)}
        rec = dict(model=name, durum="başarılı", sure_sn=round(time.time() - t, 1),
                   gpu_tepe_mib=g.peak, gpu_taban_mib=g.base, rmse=round(rmse, 2), aylar=per_m)
    except Exception as e:  # noqa: BLE001 — bilinçli: hata raporlanır, gizlenmez
        rec = dict(model=name, durum="BAŞARISIZ", sure_sn=round(time.time() - t, 1),
                   hata=f"{type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc(limit=1)
    results.append(rec)
    print(f"  {rec}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    base = F.base_table("train")
    a = base[base[config.TARGET].notna()].reset_index(drop=True)
    del base
    tr_m, va_m, _ = V.split_masks(a["MVT_MONTH"], "split_1")
    fit_m = tr_m & F.clean_mask(a).to_numpy()
    train, val = a[fit_m].copy(), a[va_m].copy()
    F.add_te(train, val, train["MVT_MONTH"].to_numpy())
    feats = F.feature_list(a.columns)
    counts = train["ADEP_mvt"].value_counts()
    apt = counts.index[0]
    print(f"hazırlık {time.time() - t0:.0f} sn | özellik {len(feats)} | split_1 eğitim satırı {len(train):,} | "
          f"en büyük havalimanı {apt}: {counts.iloc[0]:,} satır (payı {counts.iloc[0] / len(train):.1%})")

    trA, vaA = train[train["ADEP_mvt"] == apt], val[val["ADEP_mvt"] == apt]
    X_tr, X_va = F.matrix(trA, feats), F.matrix(vaA, feats)
    N_tr, num = F.numeric_array(X_tr, feats)
    N_va, _ = F.numeric_array(X_va, feats)
    y_tr = trA[config.TARGET].to_numpy(float)
    y_va = vaA[config.TARGET].to_numpy(float)
    mo_va = vaA["MVT_MONTH"].to_numpy()
    print(f"{apt}: eğitim {len(trA):,} × {len(feats)} | doğrulama {len(vaA):,}")

    res = []
    for nj in (8, 16):
        run(f"LightGBM v126 (n_jobs={nj})",
            lambda nj=nj: lgb.train({**LGB_V126, "n_jobs": nj, "random_state": 42},
                                    lgb.Dataset(X_tr, label=y_tr), num_boost_round=750).predict(X_va),
            y_va, mo_va, res)
    run("CatBoost GPU v126 (yalnızca sayısal)",
        lambda: CatBoostRegressor(iterations=650, learning_rate=0.045, depth=8, task_type="GPU", verbose=0,
                                  random_seed=42).fit(N_tr, y_tr).predict(N_va), y_va, mo_va, res)

    cat_tr = pd.concat([pd.DataFrame(N_tr, columns=num, index=trA.index), X_tr[F.CAT_V126].astype(str)], axis=1)
    cat_va = pd.concat([pd.DataFrame(N_va, columns=num, index=vaA.index), X_va[F.CAT_V126].astype(str)], axis=1)
    for mcc in (None, 1):
        extra = {} if mcc is None else {"max_ctr_complexity": mcc}
        run(f"CatBoost GPU + 5 kategorik (max_ctr_complexity={'varsayılan' if mcc is None else mcc})",
            lambda extra=extra: CatBoostRegressor(iterations=650, learning_rate=0.045, depth=8, task_type="GPU",
                                                  verbose=0, random_seed=42, **extra)
            .fit(Pool(cat_tr, y_tr, cat_features=F.CAT_V126)).predict(Pool(cat_va, cat_features=F.CAT_V126)),
            y_va, mo_va, res)
    run("XGBoost CUDA v126",
        lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=7, tree_method="hist",
                                 device="cuda", random_state=42, verbosity=0).fit(N_tr, y_tr).predict(N_va),
        y_va, mo_va, res)

    # Süre tahmini: satır sayısıyla doğrusal ölçek varsayımı (kaba)
    ok = {r["model"]: r["sure_sn"] for r in res if r["durum"] == "başarılı"}
    t_lgb = ok.get("LightGBM v126 (n_jobs=16)")
    t_cb = ok.get("CatBoost GPU v126 (yalnızca sayısal)")
    t_xgb = ok.get("XGBoost CUDA v126")
    est = {}
    if None not in (t_lgb, t_cb, t_xgb):
        scale = len(train) / len(trA)
        one = (3 * t_lgb + 2 * t_cb + 2 * t_xgb) * scale
        est = {"bir_bolme_v126_tohumlari_sn": round(one), "split1+split2_sn": round(2 * one),
               "12_ay_tam_egitim_sn": round(one * 1.2), "v126_ozgun_3kat_12ay_sn": round(3 * one * 0.8 * 1.2),
               "tek_tohum_LGB_bir_bolme_sn": round(t_lgb * scale)}
        print("\nSüre tahmini (tüm havalimanları, satır sayısıyla doğrusal varsayım):")
        for k, v in est.items():
            print(f"  {k:32s} {v:>7,} sn ≈ {v / 60:6.1f} dk")
    else:
        print("\nSüre tahmini yapılamadı: temel modellerden biri başarısız.")
    V.log_experiment({"ad": "A4_sure_bellek", "tur": "A4", "havalimani": apt, "egitim_satiri": len(trA),
                      "ozellik": len(feats), "sonuclar": res, "tahmin": est})
    print(f"toplam {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()

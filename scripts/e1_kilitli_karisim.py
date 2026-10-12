"""E1 — Kilitli satır karışımı, vekil. Tek tohumlu LightGBM, v126 özellikleri (134).

Kilit: |BLOCK − SCHED| ≤ 60 sn. BLOCK − SCHED = y − SCHED_MINUS_AOBT_v3 (y = BLOCK − AOBT_3; özellik ham zamanlarla
birebir aynı).
Havalimanı başına: (1) LightGBM sınıflandırıcı → p; (2) kilitsiz eğitim satırlarıyla LightGBM regresyon → μ.
Karışım: p·A + (1 − p)·μ, A = SCHED_MINUS_AOBT_v3; 60 sn alt sınır compare() içinde.
Taban: B1'in tek tohumlu LightGBM'i (lgb_s42; aynı özellikler ve ayarlar, tüm satırlar).
Varyantlar: (a) tüm havalimanları, (b) yalnızca LIRF (diğerleri lgb_s42).
Teşhis: havalimanı bazında AUC (tüm satırlar ve |A| > 600 olanlar), Σp / gerçek kilitli sayısı (kalibrasyon).
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, models_v126 as M, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
TOL = 60.0
A_COL = "SCHED_MINUS_AOBT_v3"
CLF_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.75,
                  bagging_fraction=0.85, bagging_freq=1, n_jobs=8, verbosity=-1, random_state=config.SEED)
CLF_ROUNDS = 400


def auc(y, p):
    """ROC AUC (Mann–Whitney); tek sınıf varsa NaN."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(split):
    t0 = time.time()
    train, val, vm = F.prepare_split(split)
    feats = F.feature_list(train.columns)
    y_tr, y_va = train[config.TARGET].to_numpy(float), val[config.TARGET].to_numpy(float)
    a_tr, a_va = train[A_COL].to_numpy(float), val[A_COL].to_numpy(float)
    lock_tr, lock_va = np.abs(y_tr - a_tr) <= TOL, np.abs(y_va - a_va) <= TOL
    p, mu = np.full(len(val), np.nan), np.full(len(val), np.nan)
    diag = {}
    print(f"[{split}] eğitim {len(train):,} (kilitli %{100 * lock_tr.mean():.2f}) | doğrulama {len(val):,} "
          f"(kilitli %{100 * lock_va.mean():.2f})", flush=True)
    for apt in sorted(val["ADEP_mvt"].unique()):
        ta = time.time()
        tr_i, va_i = (train["ADEP_mvt"] == apt).to_numpy(), (val["ADEP_mvt"] == apt).to_numpy()
        X_tr, X_va = F.matrix(train[tr_i], feats), F.matrix(val[va_i], feats)
        clf = lgb.train(CLF_PARAMS, lgb.Dataset(X_tr, label=lock_tr[tr_i].astype(float)), num_boost_round=CLF_ROUNDS)
        p[va_i] = clf.predict(X_va)
        keep = ~lock_tr[tr_i]
        mu[va_i] = M.lgb_single(X_tr[keep], y_tr[tr_i][keep], X_va)
        far = np.abs(a_va[va_i]) > 600
        lk, pp = lock_va[va_i], p[va_i]
        diag[apt] = {"kilitli": int(lk.sum()), "sigma_p": float(pp.sum()), "auc": auc(lk, pp),
                     "uzak_kilitli": int(lk[far].sum()), "uzak_sigma_p": float(pp[far].sum()),
                     "uzak_auc": auc(lk[far], pp[far])}
        d = diag[apt]
        print(f"  {apt}: kilitli {d['kilitli']:,} Σp {d['sigma_p']:.0f} AUC {d['auc']:.3f} | uzak: kilitli "
              f"{d['uzak_kilitli']:,} Σp {d['uzak_sigma_p']:.0f} AUC {d['uzak_auc']:.3f} | {time.time() - ta:.0f} sn",
              flush=True)
    mix = p * a_va + (1 - p) * mu

    b1 = val[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED_DIR / f"B1_{split}.parquet"), on="MVT_ID_mvt", how="left",
                                   validate="one_to_one")
    assert b1["lgb_s42"].notna().all() and np.allclose(b1["y"], y_va)
    days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
    nm = val["NM_TAXI_DIFF_SEC"].to_numpy(float)
    lirf = val["ADEP_mvt"].eq("LIRF").to_numpy()
    variants = {"E1_karisim_tum": mix, "E1_karisim_LIRF": np.where(lirf, mix, b1["lgb_s42"])}
    for name, pred in variants.items():
        res = V.compare(y_va, b1["lgb_s42"], pred, val["MVT_MONTH"], days, vm, nm_diff=nm)
        print(V.format_comparison(res, f"{split}: {name} ↔ lgb_s42"), flush=True)
        V.log_experiment({"ad": f"{name}_{split}", "tur": "E1", "split": split, "taban": "B1 lgb_s42",
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                    "kabul": res["accepted"]},
                          "teshis": diag, "parametreler": {"clf": CLF_PARAMS, "clf_tur": CLF_ROUNDS, "tol": TOL}})
    pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "p_kilit": p, "mu_kilitsiz": mu, "karisim": mix}
                 ).to_parquet(PRED_DIR / f"E1_{split}.parquet", index=False)
    print(f"[{split}] süre {time.time() - t0:.0f} sn", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for split in ("split_1", "split_2"):
        run(split)


if __name__ == "__main__":
    main()

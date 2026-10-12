"""v126 YENİ_200 tri-modeli (PRC_4\\src\\train_and_build_v126_leaves200.py:221-315), düzenek için.

Farklar (bilinçli):
- v126 her havalimanında 12 ay üzerinde 3 katlı rastgele KFold ile eğitip kat modellerini ortalıyordu; burada model
  bölmenin eğitim aylarının tamamıyla bir kez eğitilir. Tohumlar ve hiperparametreler aynı.
- GPU hataları yutulmaz (v126:293-308 hata olursa sessizce LightGBM tahmini koyuyordu).
"""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor

from . import features_v126 as F

LGB_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.035, num_leaves=200, max_depth=-1,
                  min_child_samples=25, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1,
                  n_jobs=8, verbosity=-1)
LGB_ROUNDS = 750
SEEDS = {"lgb": (42, 123, 777), "cb": (42, 2026), "xgb": (42, 999)}
WEIGHTS = {"cb": 0.45, "lgb": 0.40, "xgb": 0.15}


def fit_predict(X_tr, y_tr, X_pred, feats, models=("lgb", "cb", "xgb")):
    """Tek havalimanı: model başına tohum ortalaması ofset tahmini. 'lgb_s42' tek tohumlu vekildir (A2)."""
    out = {}
    if "lgb" in models:
        preds = []
        for s in SEEDS["lgb"]:
            m = lgb.train({**LGB_PARAMS, "random_state": s}, lgb.Dataset(X_tr, label=y_tr), num_boost_round=LGB_ROUNDS)
            preds.append(m.predict(X_pred))
        out["lgb_s42"], out["lgb"] = preds[0], np.mean(preds, axis=0)
    if "cb" in models or "xgb" in models:
        n_tr, _ = F.numeric_array(X_tr, feats)
        n_pr, _ = F.numeric_array(X_pred, feats)
    if "cb" in models:
        out["cb"] = np.mean([CatBoostRegressor(iterations=650, learning_rate=0.045, depth=8, task_type="GPU",
                                               verbose=0, random_seed=s).fit(n_tr, y_tr).predict(n_pr)
                             for s in SEEDS["cb"]], axis=0)
    if "xgb" in models:
        out["xgb"] = np.mean([xgb.XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=7, tree_method="hist",
                                               device="cuda", random_state=s, verbosity=0)
                              .fit(n_tr, y_tr).predict(n_pr) for s in SEEDS["xgb"]], axis=0)
    return out


def lgb_single(X_tr, y_tr, X_pred, seed=42):
    """Tek tohumlu LightGBM vekili (v126 parametreleri)."""
    m = lgb.train({**LGB_PARAMS, "random_state": seed}, lgb.Dataset(X_tr, label=y_tr), num_boost_round=LGB_ROUNDS)
    return m.predict(X_pred)


def blend(p):
    """v126 harman ağırlıkları (v126:311,315): 0,45·CB + 0,40·LGB + 0,15·XGB."""
    return WEIGHTS["cb"] * p["cb"] + WEIGHTS["lgb"] * p["lgb"] + WEIGHTS["xgb"] * p["xgb"]

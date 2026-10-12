"""Global derin CatBoost varyantları, düzenekte (A2'nin devamı; D3).

A2'de lr 0,06 ile 3.000 iterasyonun sonunda model hâlâ iyileşiyordu (en iyi iterasyon 2.998 / 2.999). Burada iterasyon
sınırı ve diğer ayarlar değiştirilebilir; erken durdurma yine eğitim günlerinden iç kümede (od_wait 200).
Kullanım: python scripts/gcb_varyant.py <etiket> <iterasyon> <lr> <depth> [kayıp] [tohum=N] [kat=ek5] [nan=1] [cb:<ad>=<değer> ...] [split ...]
kayıp: CatBoost loss_function (varsayılan RMSE; ör. Huber:delta=1000). kat / nan: girdi varyantları; cb:…: ek CatBoost
ayarları (src/gcb.py).
Tahminler data/preds/GCB_<etiket>_<split>.parquet; tek başına ve B1 harmanıyla %50 karşılaştırılır.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, gcb as G, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"


def run(tag, iters, lr, depth, split, loss="RMSE", seed=42, kat="", nan=False, extra_params=None):
    t0 = time.time()
    F.kuyruk_denetle(tag)
    train, val, vm = F.prepare_split(split)
    feats = F.feature_list(train.columns)
    cats = G.cats(kat)
    d_tr, d_va = G.frame(train, feats, kat, nan), G.frame(val, feats, kat, nan)
    y = np.clip(train[config.TARGET].to_numpy(float), -7200, 7200)
    es = V.inner_es_mask(V.day_of_year(train["MVT_MONTH"], train["MVT_DAY"]))
    params = dict(iterations=iters, learning_rate=lr, depth=depth, task_type="GPU", random_seed=seed, verbose=1000,
                  od_type="Iter", od_wait=200, use_best_model=True, loss_function=loss, **(extra_params or {}))
    model = G.fit(params, d_tr[~es], y[~es], cats, eval_set=Pool(d_tr[es], y[es], cat_features=cats))
    p = model.predict(Pool(d_va, cat_features=cats))
    b1 = val[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED_DIR / f"B1_{split}.parquet"), on="MVT_ID_mvt", how="left",
                                   validate="one_to_one")
    days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
    nm = b1["NM_TAXI_DIFF_SEC"]
    extra = {"iterasyon": model.get_best_iteration(), "parametreler": params, "kat": kat, "nan": nan,
             "sure_sn": round(time.time() - t0, 1)}
    for name, pred in (("tek", p), ("yuzde50", 0.5 * b1["blend"] + 0.5 * p)):
        res = V.compare(b1["y"], b1["blend"], pred, val["MVT_MONTH"], days, vm, nm_diff=nm)
        print(V.format_comparison(res, f"{split}: GCB_{tag} {name} ↔ B1 harmanı (en iyi iterasyon "
                                       f"{extra['iterasyon']})"), flush=True)
        V.log_experiment({"ad": f"GCB_{tag}_{name}_{split}", "tur": "D3", "split": split,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                    "kabul": res["accepted"]}, **extra})
    pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "global_cb": p}).to_parquet(
        PRED_DIR / f"GCB_{tag}_{split}.parquet", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tag, iters, lr, depth = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    rest = sys.argv[5:]
    opt = dict(a.split("=", 1) for a in rest if a.startswith(("tohum=", "kat=", "nan=")))
    loss = next((a for a in rest if not a.startswith(("split_", "tohum=", "kat=", "nan=", "cb:"))), "RMSE")
    for s in ([a for a in rest if a.startswith("split_")] or ["split_1", "split_2"]):
        run(tag, iters, lr, depth, s, loss, int(opt.get("tohum", 42)), opt.get("kat", ""), opt.get("nan", "0") == "1",
            G.cb_params(rest))

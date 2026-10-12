"""Derinlik öncelikli (Depthwise / Lossguide) global CatBoost taraması; veri her bölmede bir kez hazırlanır.

Kuyruk 15'te Depthwise (depth 9, min_data_in_leaf 50, lr 0,06) tek başına simetrik GCB 3.000 it. ile aynı düzeydeydi, ama
harmanda daha çeşitli ve ~15 kat hızlıydı (`kutuk/gcb_depthwise.log`, `kutuk/g1_H_C_dw.log`). Burada ayar taraması: her
yapılandırma iç erken durdurmayla (od_wait 200) en çok 8.000 iterasyon; hedef ±7.200'de kırpık, tohum 42.
Tahminler data/preds/GCB_<ad>_<split>.parquet; tek başına A2 GCB'ye (simetrik, 3.000 it.) ve ilk Depthwise'a karşı.
Kullanım: python scripts/gcb_dw_tarama.py [ad ...]   (ad verilmezse TARAMA'nın hepsi)
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
DW = dict(grow_policy="Depthwise", min_data_in_leaf=50)
TARAMA = {
    "dw_lr03": dict(lr=0.03, depth=9, cb=DW),
    "dw_d11": dict(lr=0.06, depth=11, cb=DW),
    "dw_d11_m200": dict(lr=0.06, depth=11, cb={**DW, "min_data_in_leaf": 200}),
    "dw_l2_10": dict(lr=0.06, depth=9, cb={**DW, "l2_leaf_reg": 10}),
    "lg_255": dict(lr=0.06, depth=12, cb=dict(grow_policy="Lossguide", max_leaves=255, min_data_in_leaf=50)),
    "dw_nan": dict(lr=0.06, depth=9, cb=DW, nan=True),
    "dw_ek5": dict(lr=0.06, depth=9, cb=DW, kat="ek5"),
    # ikinci tur (29.09 akşam): tek başına ve harmanda en iyi iki ayarın birleşimi
    "dw_lr03_ek5": dict(lr=0.03, depth=9, cb=DW, kat="ek5"),
    "dw_lr03_ek5_l2_10": dict(lr=0.03, depth=9, cb={**DW, "l2_leaf_reg": 10}, kat="ek5"),
    # 30.09: aynı ayar, kuyruk özellikleriyle (PRC_KUYRUK=1)
    "dw_lr03_ek5_l2_10_kq": dict(lr=0.03, depth=9, cb={**DW, "l2_leaf_reg": 10}, kat="ek5"),
    "dw_lr03_ek5_l2_10_kqg": dict(lr=0.03, depth=9, cb={**DW, "l2_leaf_reg": 10}, kat="ek5"),
    "dw_lr03_ek5_l2_10_kqgpy": dict(lr=0.03, depth=9, cb={**DW, "l2_leaf_reg": 10}, kat="ek5"),  # 02.10: + yüzey/kar
    "dw_lr03_ek5_l2_10_kqgp": dict(lr=0.03, depth=9, cb={**DW, "l2_leaf_reg": 10}, kat="ek5"),  # 01.10: + pist endeksi
    "dw_lr03_ek5_l2_10_kqgp_i16k": dict(lr=0.03, depth=9, cb={**DW, "l2_leaf_reg": 10}, kat="ek5", iter=16000),  # 01.10: kqgp'de en iyi iterasyon 8.000 tavanında
}
REFS = ("A2_global_cb", "GCB_depthwise")


def frame_key(name):
    return TARAMA[name].get("kat", ""), TARAMA[name].get("nan", False)


def by_id(val, path):
    return val[["MVT_ID_mvt"]].merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="one_to_one")


def main(names, splits=None):
    for n in names:
        F.kuyruk_denetle(n)
    for split in (splits or V.SPLITS):
        t0 = time.time()
        train, val, vm = F.prepare_split(split)
        feats = F.feature_list(train.columns)
        y = np.clip(train[config.TARGET].to_numpy(float), -7200, 7200)
        es = V.inner_es_mask(V.day_of_year(train["MVT_MONTH"], train["MVT_DAY"]))
        b1 = by_id(val, PRED_DIR / f"B1_{split}.parquet")
        days = V.day_of_year(val["MVT_MONTH"], val["MVT_DAY"])
        refs = {r: by_id(val, PRED_DIR / f"{r}_{split}.parquet")["global_cb"] for r in REFS
                if (PRED_DIR / f"{r}_{split}.parquet").exists()}  # split_3 için referans yok
        print(f"[{split}] hazırlık {time.time() - t0:.0f} sn", flush=True)
        for fk in dict.fromkeys(frame_key(n) for n in names):
            d_tr, d_va = G.frame(train, feats, *fk), G.frame(val, feats, *fk)
            cats = G.cats(fk[0])
            for n in [n for n in names if frame_key(n) == fk]:
                c, ta = TARAMA[n], time.time()
                params = dict(iterations=c.get("iter", 8000), learning_rate=c["lr"], depth=c["depth"], task_type="GPU", random_seed=42,
                              verbose=1000, od_type="Iter", od_wait=200, use_best_model=True, **c["cb"])
                model = G.fit(params, d_tr[~es], y[~es], cats, eval_set=Pool(d_tr[es], y[es], cat_features=cats))
                p = model.predict(Pool(d_va, cat_features=cats))
                pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "global_cb": p}).to_parquet(
                    PRED_DIR / f"GCB_{n}_{split}.parquet", index=False)
                it, sure = model.get_best_iteration(), round(time.time() - ta, 1)
                for r, rp in refs.items():
                    res = V.compare(b1["y"], rp, p, val["MVT_MONTH"], days, vm, nm_diff=b1["NM_TAXI_DIFF_SEC"])
                    print(V.format_comparison(res, f"{split}: {n} ↔ {r} (en iyi iterasyon {it}, {sure:.0f} sn)"),
                          flush=True)
                    V.log_experiment({"ad": f"GCB_{n}_{split}_vs_{r}", "tur": "D3-DW", "split": split,
                                      "parametreler": params, "kat": fk[0], "nan": fk[1], "iterasyon": it,
                                      "sure_sn": sure,
                                      "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                                "delta": res["delta_weighted_rmse"],
                                                "delta_ci": res["bootstrap"]["drmse_ci"],
                                                "gunler": f"{res['days_won']}/{res['n_days']}",
                                                "kurallar": res["rules"], "kabul": res["accepted"]}})
            del d_tr, d_va


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arg = sys.argv[1:]
    main([a for a in arg if not a.startswith("split_")] or list(TARAMA),
         [a for a in arg if a.startswith("split_")] or None)

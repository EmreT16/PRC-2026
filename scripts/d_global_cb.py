"""D3 — Global derin CatBoost, 12 ayın tamamıyla; testin Model A satırlarının tahmini.

A2'deki ayarlar: depth 9, lr 0,06, hedef ofset ±7.200'de kırpık, ADEP + 5 kategorik + v126 sayısalları (NaN → 0).
Düzenekte (A2) iç erken durdurma en iyi iterasyonu 2.998 / 2.999 buldu (3.000 sınırında); 6.000 sınırında da 5.995 /
5.998 (gcb_varyant.py it6000). Burada iterasyon sayısı sabit.
Kullanım: python scripts/d_global_cb.py [iterasyon=3000] [etiket] [tohum=42] [lr=0.06] [depth=9] [kat=ek5] [nan=1]
       [cb:<ad>=<değer> ...]
(ilk üçü sıralı; lr / depth / kat / nan anahtar=değer; kat ve nan girdi varyantları, cb:… ek CatBoost ayarları, src/gcb.py)
Tohum virgüllü liste olabilir (ör. 42,2026,7): veri bir kez hazırlanır, her tohum TEST_gcb_<etiket>_t<tohum>.parquet'e
yazılır (ortalama: gcb_ortalama.py). Tek tohumda dosya adı TEST_gcb_<etiket>.parquet.
Çıktı: data/preds/TEST_gcb.parquet ya da TEST_gcb_<etiket>.parquet (MVT_ID_mvt, NM_TAXI_DIFF_SEC, global_cb).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, gcb as G, validation as V  # noqa: E402


def main(iters=3000, tag="", seeds=(42,), lr=0.06, kat="", nan=False, extra_params=None, depth=9):
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    F.kuyruk_denetle(tag)
    train, test = F.prepare_full()
    feats = F.feature_list(train.columns)
    cats = G.cats(kat)
    d_tr, d_te = G.frame(train, feats, kat, nan), G.frame(test, feats, kat, nan)
    y = np.clip(train[config.TARGET].to_numpy(float), -7200, 7200)
    ids, nm = test["MVT_ID_mvt"].to_numpy(), test["NM_TAXI_DIFF_SEC"].to_numpy(float)
    del train, test
    for seed in seeds:
        params = dict(iterations=iters, learning_rate=lr, depth=depth, task_type="GPU", random_seed=seed, verbose=500,
                      **(extra_params or {}))
        model = G.fit(params, d_tr, y, cats)
        p = model.predict(Pool(d_te, cat_features=cats))
        name = (f"gcb_{tag}" if tag else "gcb") + (f"_t{seed}" if len(seeds) > 1 else "")
        path = config.DATA_DIR / "preds" / f"TEST_{name}.parquet"
        pd.DataFrame({"MVT_ID_mvt": ids, "NM_TAXI_DIFF_SEC": nm, "global_cb": p}).to_parquet(path, index=False)
        V.log_experiment({"ad": f"TEST_{name}", "tur": "D", "parametreler": params, "kat": kat, "nan": nan,
                          "egitim_satiri": len(y), "test_satiri": len(p), "dosya": str(path),
                          "sure_sn": round(time.time() - t0, 1)})
        print(f"bitti: {path} | ort. ofset {p.mean():.1f} | süre {time.time() - t0:.0f} sn", flush=True)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith(("lr=", "depth=", "kat=", "nan=", "cb:"))]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith(("lr=", "depth=", "kat=", "nan=")))
    main(int(pos[0]) if len(pos) > 0 else 3000, pos[1] if len(pos) > 1 else "",
         tuple(int(t) for t in pos[2].split(",")) if len(pos) > 2 else (42,),
         float(opt.get("lr", 0.06)), opt.get("kat", ""), opt.get("nan", "0") == "1", G.cb_params(sys.argv[1:]),
         int(opt.get("depth", 9)))

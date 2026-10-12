"""09.10 akşam: LTFM satırlarında havalimanı başına LightGBM ayar taraması.

Neden: LTFM Model A SSE'sinin %18'i (2025 Oca / Tem OOF, RMSE 226), ADS-B / OPDI yok → ADS-B düzelticisi çalışmıyor; kazanç
ancak Model A'dan. lgbh3 / E1 / XGB havalimanı başına eğitiliyor (src/aday.py fit_predict_airport) → yalnız LTFM için farklı
kayıp / kapasite seçilebilir. Bütün havalimanları için tek ayar (Huber δ 1.000, 200 yaprak, 750 tur) seçilmişti.
Düzenek: 6 bölme (her biri iki ay dışarıda; 12 ayın hepsi bir kez doğrulama), yalnız LTFM satırları, kqgp özellikleri
(PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 — harmanın OOF'u ADAY_v_lgb_huber1000_3t_kqgp ile aynı aile), tek tohum (42)
tarama; değişkenler aynı tohumla kıyaslanır. Ölçü taksi ölçeğinde RMSE (60 sn alt sınır), bölme ve ay başına.
Kullanım: PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 python scripts/ltfm_tarama.py [degiskenler=...] [bolmeler=1,2,...]
Çıktı: kutuk/ltfm_tarama.log, data/preds/LTFM_tarama_<değişken>_split_k.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import aday as AD, config, features_v126 as F  # noqa: E402

TABAN = dict(AD.ADAYLAR["v_lgb_huber1000_3t_kqgp"], lgb_tohum=(42,))
DEGISKEN = {
    "taban": {},
    "h600": dict(lgb_huber=600),
    "h2000": dict(lgb_huber=2000),
    "l2": dict(lgb_huber=None),
    "y63": dict(lgb_ek=dict(num_leaves=63)),
    "y400": dict(lgb_ek=dict(num_leaves=400)),
    "m100": dict(lgb_ek=dict(min_child_samples=100)),
    "lr02": dict(lgb_ek=dict(learning_rate=0.02), lgb_tur=1300),
}
APT = "LTFM"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    adlar = opt.get("degiskenler", ",".join(DEGISKEN)).split(",")
    bolmeler = [f"split_{k}" for k in opt.get("bolmeler", "1,2,3,4,5,6").split(",")]
    sonuc = []
    for sp in bolmeler:
        t0 = time.time()
        train, val, vm = F.prepare_split(sp)
        spec0 = dict(TABAN)
        train, val = AD.kuyruk_ekle(spec0, train, val)
        feats = AD.features(spec0, train.columns)
        tr, va = train[train.ADEP_mvt == APT], val[val.ADEP_mvt == APT]
        nm, y = va["NM_TAXI_DIFF_SEC"].to_numpy(float), va[config.TARGET].to_numpy(float)
        yt = nm - y
        for ad in adlar:
            spec = {**TABAN, **DEGISKEN[ad]}
            out = AD.fit_predict_airport(spec, tr, va, feats)
            p = np.maximum(60.0, nm - out["harman"])
            pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "ofset": out["harman"]}).to_parquet(
                config.DATA_DIR / "preds" / f"LTFM_tarama_{ad}_{sp}.parquet", index=False)
            for m in sorted(va.MVT_MONTH.unique()):
                i = (va.MVT_MONTH == m).to_numpy()
                sonuc.append(dict(bolme=sp, ay=int(m), degisken=ad, n=int(i.sum()), sse=float(np.sum((p[i] - yt[i]) ** 2))))
            print(f"{sp} {ad:<6} RMSE {np.sqrt(np.mean((p - yt) ** 2)):7.2f}", flush=True)
        print(f"{sp} süre {time.time() - t0:.0f} sn", flush=True)
    s = pd.DataFrame(sonuc)
    s.to_csv(config.LOG_DIR / "ltfm_tarama_sonuc.csv", index=False)
    t = s.pivot_table(index="ay", columns="degisken", values="sse", aggfunc="sum")
    n = s[s.degisken == adlar[0]].groupby("ay").n.sum()
    rm = np.sqrt(t.div(n, axis=0))
    print("\nLTFM ay başına RMSE (taksi, tek tohum):")
    print(rm[adlar].round(2).to_string())
    top = np.sqrt(t.sum() / n.sum())
    print("\ntoplam RMSE:", top[adlar].round(2).to_dict())
    if "taban" in adlar:
        iyi = (rm.lt(rm["taban"], axis=0)).sum()
        print("tabandan iyi ay sayısı (/12):", iyi[adlar].to_dict())


if __name__ == "__main__":
    main()

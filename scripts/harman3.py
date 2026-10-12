"""Üç bölmeli harman ağırlığı (30.09): bileşen adları src/adsb.py BILESEN anahtarları (g_test_harman.py MAP'te de olmalı).

Her bölme sırayla dışarıda: ağırlıklar diğer iki bölmenin birleşik OOF'unda SLSQP ile seçilir (negatif olmayan, toplamı 1;
ay ağırlıklı ofset MSE), dışarıdaki bölmede taksi RMSE'si (60 sn alt sınırlı) ölçülür. Karşılaştırma: aynı bileşenlerle
sabit ağırlık (verilirse, --taban=<json>). Önerilen ağırlık: üç bölmenin tamamında seçilen.
Kullanım: python scripts/harman3.py <etiket> <bileşen> [...] [--taban=kutuk/g1_agirliklar_kq.json]
Çıktı: kutuk/g1_agirliklar_<etiket>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config, validation as V  # noqa: E402

PRED = config.DATA_DIR / "preds"


def yukle(split, adlar):
    d = pd.read_parquet(PRED / f"B1_{split}.parquet", columns=["MVT_ID_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y"])
    for ad in adlar:
        f, col = A.BILESEN[ad]
        b = pd.read_parquet(PRED / f"{f}_{split}.parquet", columns=["MVT_ID_mvt", col]).rename(columns={col: ad})
        d = d.merge(b, on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert d[adlar].notna().all().all(), split
    return d


def sec(d, adlar):
    aylar = sorted(d["MVT_MONTH"].unique())
    rw = d["MVT_MONTH"].map({m: 1.0 / (d["MVT_MONTH"] == m).sum() for m in aylar}).to_numpy(float)
    P, y = d[adlar].to_numpy(float), d["y"].to_numpy(float)
    res = minimize(lambda a: float(np.sum(rw * (P @ a - y) ** 2)), np.full(len(adlar), 1 / len(adlar)), method="SLSQP",
                   bounds=[(0, 1)] * len(adlar), constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}])
    return np.round(res.x, 4)


def rmse_taksi(d, adlar, a):
    nm = d["NM_TAXI_DIFF_SEC"].to_numpy(float)
    p = np.maximum(V.TAXI_FLOOR, nm - d[adlar].to_numpy(float) @ a)
    y = nm - d["y"].to_numpy(float)
    return {int(m): float(np.sqrt(np.mean((p[d["MVT_MONTH"] == m] - y[d["MVT_MONTH"] == m]) ** 2)))
            for m in sorted(d["MVT_MONTH"].unique())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    etiket = sys.argv[1]
    adlar = [a for a in sys.argv[2:] if not a.startswith("--")]
    taban = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--taban=")), None)
    data = {s: yukle(s, adlar) for s in V.SPLITS}
    tw = None
    if taban:
        g = json.loads(Path(taban).read_text(encoding="utf-8"))
        tw = np.array([dict(zip(g["bilesenler"], g["agirlik"])).get(a, 0.0) for a in adlar])
    for s in V.SPLITS:
        diger = pd.concat([data[o] for o in V.SPLITS if o != s], ignore_index=True)
        a = sec(diger, adlar)
        r = rmse_taksi(data[s], adlar, a)
        satir = f"{s} dışarıda: ağırlık {dict(zip(adlar, a.tolist()))} → " + ", ".join(f"ay {m} {v:.2f}" for m, v in r.items())
        if tw is not None:
            r0 = rmse_taksi(data[s], adlar, tw)
            satir += " | sabit: " + ", ".join(f"ay {m} {v:.2f}" for m, v in r0.items())
        print(satir, flush=True)
    a = sec(pd.concat(data.values(), ignore_index=True), adlar)
    yol = config.ROOT / "kutuk" / f"g1_agirliklar_{etiket}.json"
    yol.write_text(json.dumps({"bilesenler": adlar, "agirlik": a.tolist(),
                               "not": "harman3.py: üç bölmenin birleşik OOF'unda seçildi"}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"önerilen (3 bölme): {dict(zip(adlar, a.tolist()))} → {yol}")


if __name__ == "__main__":
    main()

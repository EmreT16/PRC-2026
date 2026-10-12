"""D — 12 ayın tamamıyla eğitim ve testin Model A satırlarının tahmini (D1).

Kullanım: python scripts/d_tam_egitim.py <aday>
Tur sayıları v126'nın sabit değerleri (düzenekte B1 ile aynı). Aday tanımları src/aday.py (ADAYLAR).
Çıktı: data/preds/TEST_<aday>.parquet (MVT_ID_mvt, NM_TAXI_DIFF_SEC, bileşen ofsetleri, harman ofseti,
taksi = max(60, NM_TAXI_DIFF − harman)).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import aday as AD, config, features_v126 as F, validation as V  # noqa: E402


def main(name):
    spec = AD.ADAYLAR[name]
    t0 = time.time()
    assert not F.KUYRUK or "_kq" in name, "PRC_KUYRUK=1 iken aday adı '_kq' içermeli"
    assert F.GECIKME == ("_kqg" in name), "PRC_GECIKME=1 ile aday adındaki '_kqg' uyuşmalı"
    assert F.GECPIST == ("_kqgp" in name), "PRC_GECPIST=1 ile aday adındaki '_kqgp' uyuşmalı"
    assert F.EKOZ == ("_kqgpy" in name), "PRC_EKOZ=1 ile aday adındaki '_kqgpy' uyuşmalı"
    assert F.GUNSUZ == ("gunsuz" in name), "PRC_GUNSUZ=1 ile aday adındaki 'gunsuz' uyuşmalı"
    train, test = F.prepare_full()
    train, test = AD.kuyruk_ekle(spec, train, test)
    feats = AD.features(spec, train.columns)
    print(f"[{name}] eğitim {len(train):,} | test {len(test):,} | özellik {len(feats)} | hazırlık "
          f"{time.time() - t0:.0f} sn", flush=True)
    preds = {}
    for apt in sorted(test["ADEP_mvt"].unique()):
        ta = time.time()
        tr_i, te_i = (train["ADEP_mvt"] == apt).to_numpy(), (test["ADEP_mvt"] == apt).to_numpy()
        out = AD.fit_predict_airport(spec, train[tr_i], test[te_i], feats)
        for k, v in out.items():
            preds.setdefault(k, np.full(len(test), np.nan))[te_i] = v
        print(f"  {apt}: eğitim {tr_i.sum():,} test {te_i.sum():,} | {time.time() - ta:.0f} sn", flush=True)
    nm = test["NM_TAXI_DIFF_SEC"].to_numpy(float)
    taksi = np.maximum(V.TAXI_FLOOR, nm - preds["harman"])
    assert np.isfinite(taksi).all()
    out = pd.DataFrame({"MVT_ID_mvt": test["MVT_ID_mvt"].to_numpy(), "NM_TAXI_DIFF_SEC": nm, **preds, "taksi": taksi})
    path = config.DATA_DIR / "preds" / f"TEST_{name}.parquet"
    out.to_parquet(path, index=False)
    V.log_experiment({"ad": f"TEST_{name}", "tur": "D", "tanim": spec, "ozellik_sayisi": len(feats),
                      "egitim_satiri": len(train), "test_satiri": len(test), "dosya": str(path),
                      "taksi_ort": float(taksi.mean()), "sure_sn": round(time.time() - t0, 1)})
    print(f"[{name}] bitti: {path} | ort. taksi {taksi.mean():.1f} | süre {time.time() - t0:.0f} sn", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])

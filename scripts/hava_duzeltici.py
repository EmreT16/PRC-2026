"""İzsiz satırlar için hava durumlu kalıntı düzeltici (deney): Model A (D_dw) tabanına görüş mesafesi / LVP ile.

Neden: Model A'nın 134 özelliğinde VISIBILITY, IS_LVP, IS_LOW_VIS yok (base_v126 tablosunda var).
Görüş mesafesi tek bilgi taşıyan hava değişkeni, düşük görüş prosedürleri tüm yer hareketini yavaşlatıyor
(kaybı 4,5 sn). ADS-B izi olan satırlarda gerçek hareket zaten görülüyor; burada yalnızca izsiz satırlar (LTFM, LFPG, ...).
Doğrulama: Ocak↔Temmuz (eğitimde karşı ay + Şubat + Ağustos), yalnız izsiz satırlarda ölçülür.
Kullanım: python scripts/hava_duzeltici.py cv
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_duzeltici as D  # noqa: E402
from src import config, validation as V  # noqa: E402

HAVA = ["VISIBILITY", "IS_LVP", "IS_LOW_VIS", "TEMPERATURE", "DEW_POINT", "TEMP_DEWPOINT_SPREAD", "SNOWFALL",
        "SNOWFALL_3H", "PRECIP_3H", "WIND_SPEED", "CLOUD_COVER"]
PARAM = dict(objective="huber", alpha=600.0, learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=8)


def cv():
    sys.stdout.reconfigure(encoding="utf-8")
    d = D.veri(True)
    h = pd.read_parquet(config.DATA_DIR / "base_v126_train_v1.parquet", columns=["MVT_ID_mvt", "MVT_HOUR"] + HAVA
                        if "MVT_HOUR" in pd.read_parquet(config.DATA_DIR / "base_v126_train_v1.parquet").columns[:0]
                        else ["MVT_ID_mvt"] + HAVA)
    d = d.merge(h, on="MVT_ID_mvt", how="left", validate="one_to_one")
    es = d["ADSB_ESLESME"].fillna(0).to_numpy() > 0
    X = pd.DataFrame({"taban": d["taban"], "nm": d["NM_TAXI_DIFF_SEC"], "apt": d["ADEP_mvt"].astype("category")})
    for c in HAVA:
        X[c] = d[c].astype(float)
    r = (d["y_taksi"] - d["taban"]).to_numpy(float)
    y, base, ay = d["y_taksi"].to_numpy(float), d["taban"].to_numpy(float), d["MVT_MONTH"].to_numpy()
    rhat = np.zeros(len(d))
    for b in (7, 1):
        tr, te = (ay != b), (ay == b) & ~es
        m = lgb.train(PARAM, lgb.Dataset(X[tr], r[tr]), 600)
        rhat[te] = m.predict(X[te])
        imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
        print(f"ay {b} sınaması: eğitim {tr.sum():,}, izsiz sınama {te.sum():,} | önemli: {', '.join(imp.index[:6])}")
    for k in (0.5, 1.0):
        p = np.maximum(60.0, base + k * rhat)
        for b in (1, 7):
            mm = (ay == b) & ~es
            print(f"k={k} ay {b} izsiz: taban {np.sqrt(np.mean((base[mm] - y[mm]) ** 2)):.2f} → "
                  f"{np.sqrt(np.mean((p[mm] - y[mm]) ** 2)):.2f}")
    t = pd.DataFrame({"apt": d["ADEP_mvt"], "ay": ay, "e0": (base - y) ** 2,
                      "e1": (np.maximum(60.0, base + rhat) - y) ** 2})[~es & np.isin(ay, (1, 7))]
    g = t.groupby(["apt", "ay"]).agg(n=("e0", "size"), taban=("e0", "mean"), yeni=("e1", "mean"))
    g[["taban", "yeni"]] = np.sqrt(g[["taban", "yeni"]])
    print(g[g.n > 500].round(1).to_string())


if __name__ == "__main__":
    {"cv": cv}[sys.argv[1]]()

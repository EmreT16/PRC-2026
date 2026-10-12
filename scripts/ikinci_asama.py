"""İkinci aşama kalıntı modeli (30.09): Model A tabanı (D_dw, 6 aylık OOF) üzerine kuyruk / pencere özellikleriyle düzeltme.
Amaç: ADS-B izi olmayan satırlar (testin ~%31'i) dahil tüm NM'li satırlarda Model A'nın görmediği bilgiyi sınamak.
Doğrulama: 6 ay (1, 2, 6, 7, 8, 12), her ay sırayla dışarıda; rapor Oca/Tem (öncekilerle karşılaştırılabilir) ve tüm aylar.
Kullanım: python scripts/ikinci_asama.py cv [oz=kuyruk|yok] [kayip=huber|l2] [tur=600] [p:num_leaves=63]
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_duzeltici as D  # noqa: E402
from src import config  # noqa: E402

PARAM = dict(objective="huber", alpha=600.0, learning_rate=0.03, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=8)
TUR = 600
KQ = ["KQ_TO_APT_PEN", "KQ_VARIS_PEN", "KQ_ONDE_APT", "KQ_ITME_20", "KQ_TO_PIST_PEN", "KQ_ONDE_PIST", "KQ_ONCEKI_PIST",
      "KQ_SONRAKI_PIST", "KQ_PIST_HIZ", "KQ_SERBEST"]


def veri():
    from src import adsb as A
    assert not A.IKINCI, "ikinci aşamanın kendisi PRC_IKINCI kapalıyken eğitilmeli"
    d = D.veri(True)
    d = d.merge(pd.read_parquet(config.DATA_DIR / "kuyruk_oz.parquet"), on="MVT_ID_mvt", how="left", validate="one_to_one")
    for ek in ("gecikme_oz.parquet", "wx_oz.parquet"):
        if (config.DATA_DIR / ek).exists():
            d = d.merge(pd.read_parquet(config.DATA_DIR / ek), on="MVT_ID_mvt", how="left", validate="one_to_one")
    if (config.DATA_DIR / "kuyruk_oz2.parquet").exists():
        d = d.merge(pd.read_parquet(config.DATA_DIR / "kuyruk_oz2.parquet"), on="MVT_ID_mvt", how="left",
                    validate="one_to_one")
    return d


def matris(d, oz):
    X = pd.DataFrame({"taban": d["taban"].to_numpy(float), "nm": d["NM_TAXI_DIFF_SEC"].to_numpy(float),
                      "taban_std": d["taban_std"].to_numpy(float)})
    X["apt"] = pd.Categorical(d["ADEP_mvt"].astype(str).to_numpy())
    if oz in ("gec", "kuyruk_gec", "gec_wx", "gec_wx2", "gec_wx3"):
        for c in [c for c in d.columns if c.startswith("GEC_")]:
            X[c] = d[c].to_numpy(float)
    if oz in ("gec_wx", "wx"):
        for c in [c for c in d.columns if c.startswith(("WX_", "NM_YAPAY", "GUN_ORAN"))]:
            X[c] = d[c].to_numpy(float)
    if oz == "gec_wx3":  # yalnız olay bayrakları + yapay NM (sürekli hava değişkenleri ERA5'te zaten var)
        for c in [c for c in d.columns if c.startswith(("WX_TS", "WX_FZ", "WX_SN", "WX_FG", "WX_RAY", "NM_YAPAY"))]:
            X[c] = d[c].to_numpy(float)
    if oz == "gec_wx2":  # gün oranı hariç (ay içi desen ezberi şüphesi)
        for c in [c for c in d.columns if c.startswith(("WX_", "NM_YAPAY"))]:
            X[c] = d[c].to_numpy(float)
        X["kendi_gec"] = (d["mvt"] - 0).to_numpy(float) * 0 if False else np.nan
        X = X.drop(columns=["kendi_gec"])
    if oz in ("kuyruk", "kuyruk2", "kuyruk_gec"):
        for c in KQ + ([c for c in d.columns if c.startswith("KQ2_")] if oz == "kuyruk2" else []):
            X[c] = d[c].to_numpy(float)
        X["nm_eksi_taban"] = X["nm"] - X["taban"]
    return X


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cv(opt):
    t0 = time.time()
    d = veri()
    X = matris(d, opt.get("oz", "kuyruk"))
    y, base = d["y_taksi"].to_numpy(float), d["taban"].to_numpy(float)
    r = y - base
    ay = d["MVT_MONTH"].to_numpy()
    es = d["ADSB_ESLESME"].fillna(0).to_numpy() > 0
    rhat = np.zeros(len(d))
    for b in sorted(set(ay.tolist())):
        tr, te = ay != b, ay == b
        m = lgb.train(PARAM, lgb.Dataset(X[tr], r[tr]), TUR)
        rhat[te] = m.predict(X[te])
        if b == 7:
            imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
            print("önem (Temmuz katı):", ", ".join(f"{k} {v / imp.sum():.2f}" for k, v in imp.head(8).items()))
    rows = {}
    for k in (0.5, 1.0):
        p = np.maximum(60.0, base + k * rhat)
        row = {}
        for b in sorted(set(ay.tolist())):
            mm = ay == b
            row[f"{b}"] = rmse(p[mm], y[mm])
        for ad, mm in (("Oca+Tem izsiz", np.isin(ay, (1, 7)) & ~es), ("Oca+Tem izli", np.isin(ay, (1, 7)) & es),
                       ("6 ay izsiz", ~es), ("6 ay izli", es)):
            row[ad] = rmse(p[mm], y[mm])
        rows[f"k={k}"] = row
    p0 = base
    row = {f"{b}": rmse(p0[ay == b], y[ay == b]) for b in sorted(set(ay.tolist()))}
    for ad, mm in (("Oca+Tem izsiz", np.isin(ay, (1, 7)) & ~es), ("Oca+Tem izli", np.isin(ay, (1, 7)) & es),
                   ("6 ay izsiz", ~es), ("6 ay izli", es)):
        row[ad] = rmse(p0[mm], y[mm])
    rows["taban"] = row
    print(pd.DataFrame(rows).T.round(2).to_string())
    k1 = np.maximum(60.0, base + rhat)
    g = pd.DataFrame({"apt": d["ADEP_mvt"], "e0": (base - y) ** 2, "e1": (k1 - y) ** 2, "iz": es})
    t = g.groupby(["apt", "iz"]).agg(n=("e0", "size"), taban=("e0", "mean"), yeni=("e1", "mean"))
    t[["taban", "yeni"]] = np.sqrt(t[["taban", "yeni"]])
    print(t.round(1).unstack("iz").to_string())
    ad = opt.get("ad", opt.get("oz", "kuyruk"))
    pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "rhat": rhat}).to_parquet(
        config.DATA_DIR / "preds" / f"IKINCI_cv_{ad}.parquet", index=False)
    print(f"süre {time.time() - t0:.0f} sn")


def uretim(opt):
    """6 ayın tüm OOF satırlarıyla eğit; testin Model A satırlarına kalıntı tahmini → TEST_IKINCI_<ad>.parquet.
    Test tabanı: TEST_<A.TEST_AD>.parquet (PRC_HARMAN ile seçilen harman)."""
    from src import adsb as A
    d = veri()
    oz = opt.get("oz", "gec")
    X = matris(d, oz)
    r = (d["y_taksi"] - d["taban"]).to_numpy(float)
    bil = A.bilesenler()
    t = pd.read_parquet(A.PRED / f"TEST_{A.TEST_AD}.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "taksi"] + bil)
    t["taban_std"] = t[bil].std(axis=1, ddof=0)
    t["taban"] = t["taksi"]
    t = t.merge(pd.read_parquet(config.DEP_RANK, columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt", how="left", validate="one_to_one")
    for f in ("kuyruk_oz.parquet", "gecikme_oz.parquet", "wx_oz.parquet"):
        t = t.merge(pd.read_parquet(config.DATA_DIR / f), on="MVT_ID_mvt", how="left", validate="one_to_one")
    Xt = matris(t, oz)
    Xt["apt"] = pd.Categorical(t["ADEP_mvt"].astype(str).to_numpy(), categories=X["apt"].cat.categories)
    assert list(Xt.columns) == list(X.columns)
    rhat = np.mean([lgb.train(dict(PARAM, seed=s), lgb.Dataset(X, r), TUR).predict(Xt) for s in (42, 2026, 7)], axis=0)
    ad = opt.get("ad", oz)
    pd.DataFrame({"MVT_ID_mvt": t["MVT_ID_mvt"], "rhat": rhat}).to_parquet(config.DATA_DIR / "preds" / f"TEST_IKINCI_{ad}.parquet", index=False)
    print(f"test {len(t):,} | rhat ort {rhat.mean():.1f}, std {rhat.std():.1f} | OOF rhat std "
          f"{pd.read_parquet(config.DATA_DIR / 'preds' / f'IKINCI_cv_{ad}.parquet').rhat.std():.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    if opt.get("kayip") == "l2":
        PARAM.update(objective="l2")
    for a, v in opt.items():
        if a.startswith("p:"):
            PARAM[a[2:]] = type(PARAM.get(a[2:], 0.0))(v)
    TUR = int(opt.get("tur", TUR))
    {"cv": cv, "uretim": uretim}[sys.argv[1]](opt)

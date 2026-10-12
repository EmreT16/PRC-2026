"""ADS-B düzeltici — Model B (NM'siz) satırları, EHAM/EDDF/LSZH/LEBL/EDDM.

Taban: testte kullanılan Model B tahmini. EHAM/LSZH/EDDF'te v126_B; EDDM/LEBL'de tüm satırlı model (tum6M); CV'de karşılıkları
BM_cv_oof.parquet (v126_B) ve BM_cv_tum_oof.parquet (tum_mu_yt_tavan3500). Bu havalimanlarında yer tutucu (hedef ≈ gecikme)
payı %3,2; park yerinde görülen satırlarda ADS-B'nin medyan hatası 45 sn (Model B ~151 sn), ama aykırı değerleri çok.
Seçenekler (Ocak↔Temmuz 2025 CV; yalnızca bu beş havalimanının B satırları):
  b    : yalnızca B satırlarıyla eğitilen Huber düzeltici
  a    : Model A satırlarıyla eğitilen düzelticinin B satırlarına uygulanması (nm = NaN)
  ab   : A + B satırlarıyla ortak eğitim (B'de nm = NaN, IS_B işareti)
Kullanım: python scripts/adsb_b_duzeltici.py cv
          python scripts/adsb_b_duzeltici.py uretim tur=<b|a|ab> [k=1]  → data/preds/TEST_B_<ad>.parquet (g_gonderim b=)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_duzeltici as D  # noqa: E402
from src import adsb as A, config, model_b as MB, validation as V  # noqa: E402

A5 = ["EHAM", "EDDF", "LSZH", "LEBL", "EDDM"]
V126_APT = ["EHAM", "LSZH", "EDDF"]  # testte v126_B kalan havalimanları (tum6M: EDDM, LEBL)
PARAM = dict(D.PARAM, objective="huber", alpha=600.0)


def b_veri():
    tr = MB.load_b("train")
    tr = tr[tr["MVT_MONTH"].isin([1, 7]) & tr["ADEP_mvt"].isin(A5)].reset_index(drop=True)
    oof = pd.read_parquet(config.DATA_DIR / "preds" / "BM_cv_oof.parquet", columns=["MVT_ID_mvt", "v126_B"])
    tum = pd.read_parquet(config.DATA_DIR / "preds" / "BM_cv_tum_oof.parquet",
                          columns=["MVT_ID_mvt", "tum_mu_yt_tavan3500"])
    d = tr.merge(oof, on="MVT_ID_mvt", how="left").merge(tum, on="MVT_ID_mvt", how="left")
    d = d.merge(A.ozellikler(), on="MVT_ID_mvt", how="left", validate="one_to_one")
    d["taban"] = np.where(d["ADEP_mvt"].isin(V126_APT), d["v126_B"], d["tum_mu_yt_tavan3500"])
    d["y_taksi"] = d[MB.TARGET_B].astype(float)
    d["NM_TAXI_DIFF_SEC"] = np.nan
    return d


def matris(d):
    X = D.matris(d, havalimani=False)
    X["gecikme"] = d[MB.DELAY].astype(float).to_numpy()
    X["dG"] = (d["ADSB_HAREKET"] - d[MB.DELAY]).to_numpy(float)  # ADS-B taksisi gecikmeye ne kadar yakın (yer tutucu)
    return X


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cv(opt):
    b = b_veri()
    a = D.veri()
    es_b = b["ADSB_ESLESME"].fillna(0).to_numpy() > 0
    es_a = a["ADSB_ESLESME"].fillna(0).to_numpy() > 0
    Xb = matris(b)
    Xa = matris(a.assign(**{MB.DELAY: np.nan}) if MB.DELAY not in a else a)
    Xa["gecikme"] = np.nan
    Xa["dG"] = np.nan
    rb = (b["y_taksi"] - b["taban"]).to_numpy(float)
    ra = (a["y_taksi"] - a["taban"]).to_numpy(float)
    ay_b, ay_a = b["MVT_MONTH"].to_numpy(), a["MVT_MONTH"].to_numpy()
    tahmin = {k: np.zeros(len(b)) for k in ("b", "a", "ab")}
    for tr_ay, te_ay in ((1, 7), (7, 1)):
        te = es_b & (ay_b == te_ay)
        trb, tra = es_b & (ay_b == tr_ay), es_a & (ay_a == tr_ay)
        mb = lgb.train(PARAM, lgb.Dataset(Xb[trb], rb[trb]), 300)
        tahmin["b"][te] = mb.predict(Xb[te])
        ma = lgb.train(PARAM, lgb.Dataset(Xa[tra], ra[tra]), D.TUR)
        tahmin["a"][te] = ma.predict(Xb[te])
        Xab = pd.concat([Xa[tra].assign(IS_B=0.0), Xb[trb].assign(IS_B=1.0)], ignore_index=True)
        w = np.r_[np.ones(tra.sum()), np.full(trb.sum(), float(opt.get("agirlik", 20)))]
        mab = lgb.train(PARAM, lgb.Dataset(Xab, np.r_[ra[tra], rb[trb]], weight=w), D.TUR)
        tahmin["ab"][te] = mab.predict(Xb[te].assign(IS_B=1.0))
    y, base = b["y_taksi"].to_numpy(float), b["taban"].to_numpy(float)
    rows = {}
    for ad, r in [("taban", np.zeros(len(b)))] + list(tahmin.items()):
        for k in (0.5, 1.0) if ad != "taban" else (1.0,):
            p = np.maximum(60.0, base + k * r)
            row = {f"ay {m}": rmse(p[ay_b == m], y[ay_b == m]) for m in (1, 7)}
            row["ADS-B satırı"] = rmse(p[es_b], y[es_b])
            for apt in A5:
                i = b["ADEP_mvt"].eq(apt).to_numpy()
                row[apt] = rmse(p[i], y[i])
            rows[f"{ad} k={k}"] = row
    print(f"B satırı {len(b):,} (ADS-B {es_b.sum():,}); A eğitim satırı {es_a.sum():,}")
    print(pd.DataFrame(rows).T.round(1).to_string())


def uretim(opt):
    """'a' seçeneği: 2025 Model A satırlarıyla eğitilen düzeltici (adsb_duzeltici uretim ile aynı ayar, 3 tohum) testteki
    beş havalimanının B satırlarına uygulanır. Taban = gönderilmiş v153'ün Model B taksisi (v126 + tum6M + bant).
    Çıktı: TEST_B_<ad>.parquet — v153'ün TÜM B satırları (düzeltilmeyenler aynen), g_gonderim.py b=<ad> için."""
    k = float(opt.get("k", 1.0))
    a = D.veri()
    es_a = a["ADSB_ESLESME"].fillna(0).to_numpy() > 0
    Xa = D.matris(a, havalimani=False)
    ra = (a["y_taksi"] - a["taban"]).to_numpy(float)
    modeller = [lgb.train(dict(PARAM, seed=s), lgb.Dataset(Xa[es_a], ra[es_a]), D.TUR) for s in (42, 2026, 7)]
    rk = MB.load_b("rank")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"]]
    # taban gönderim: v153 (bant kuralı dahil) ya da taban=v152 (30.09: bant kuralı LB'de +0,74 kaybettirdi → v152)
    kaynak = pd.read_parquet(config.ROOT / "gonderim" / f"strong-moose_{opt.get('taban', 'v153')}.parquet")
    t = rk.merge(kaynak, on="MVT_ID_mvt", how="left", validate="one_to_one").rename(columns={"TAXITIME_SEC_mvt": "taban"})
    t = t.merge(A.ozellikler(), on="MVT_ID_mvt", how="left", validate="one_to_one")
    t["NM_TAXI_DIFF_SEC"] = np.nan
    et = (t["ADSB_ESLESME"].fillna(0).to_numpy() > 0) & t["ADEP_mvt"].isin(A5).to_numpy()
    Xt = D.matris(t, havalimani=False)
    r = np.zeros(len(t))
    r[et] = np.mean([m.predict(Xt[et]) for m in modeller], axis=0)
    t["taksi"] = np.maximum(60.0, t["taban"] + k * r)
    t["rhat"] = r
    ad = opt.get("ad", "tum6M_bant_adsb")
    t[["MVT_ID_mvt", "ADEP_mvt", "taban", "taksi", "rhat"]].to_parquet(config.DATA_DIR / "preds" / f"TEST_B_{ad}.parquet",
                                                                     index=False)
    f = t["taksi"] - t["taban"]
    print(f"test B satırı {len(t):,} | düzeltilen {int((f != 0).sum()):,} | RMS fark {np.sqrt(np.mean(f[f != 0] ** 2)):.0f} "
          f"| ort. fark {f[f != 0].mean():.1f}")
    print(t.assign(f=f).groupby("ADEP_mvt")["f"].agg(["size", lambda s: int((s != 0).sum()), "mean"]).round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    opt = dict(x.split("=", 1) for x in sys.argv[2:] if "=" in x)
    D.PARAM.update(objective="huber", alpha=600.0)
    if opt.get("apt") == "genis":  # LIRF hariç tüm havalimanları (LIRF B satırlarında 24 saat / yer tutucu rejimleri)
        A5 = ["EHAM", "EDDF", "LSZH", "LEBL", "EDDM", "EGLL", "LEMD", "LFPG", "LTFM"]
    {"cv": cv, "uretim": uretim}[sys.argv[1]](opt)

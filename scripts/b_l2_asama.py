"""Model B için L2 ikinci aşama (08.10; Model A'daki G2 / G2b'nin B karşılığı).

Neden: A tarafında Huber eğitilmiş katmanların üstüne CatBoost L2 kalıntı aşaması SSE'yi −%1,1 düşürdü (G2 + G2b). Model B
melezinin (μ = 0,5 · LightGBM + 0,5 · CatBoost; p24 / yt / d park + ktah modelinden) kalıntısında da bağlama (gecikme, saat,
park dizisi, kopyalama geçmişi) bağlı sistematik sapma kalmış olabilir.
Ne: melez OOF tahmininin (b_model_cv_tum.py çeyrek CV'leri) kalıntısı y − p, 24 saat satırları hariç, yeni tabanlı
havalimanlarında (8: 6 + EHAM / LSZH) CatBoost L2 (derinlik 4, 600 yineleme, lr 0,03, L2 10, tohum 1 / 2 / 3) ile öğrenilir.
Sınama: çeyrek dışarıda (V.quarter_folds); ölçü 6 ve 8 havalimanı, 24 sa hariç / dahil, çeyrek ve havalimanı kırılımı.
Kullanım: python scripts/b_l2_asama.py cv              → yalnız CV
          python scripts/b_l2_asama.py uretim <ek>     → TEST_B_tum_<ek>.parquet'e uygulanır → TEST_B_tum_<ek>l2.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

APT8 = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM", "EHAM", "LSZH"]
APT6 = APT8[:6]
TOHUM = (1, 2, 3)
PB = ["PB_ON1", "PB_ON1_HY", "PB_AYNI", "PB_SON1", "PD_ON1", "PD_SON1", "PB_KT15", "PB_KT15_P"]
YTE = ["YTE_UCUS", "YTE_HY"]
F = ["p", "mu", "yt", "p24", "d", "MVT_HOUR_UTC", "MVT_DAYOFWEEK", "apt"] + PB + YTE


def ozellik(df, c):
    """df: MVT_ID_mvt, ADEP_mvt (satır sırası c ile aynı); c: bileşen sözlüğü (mu, p24, yt_tum, d, d24)."""
    X = pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt.to_numpy(), "apt": df.ADEP_mvt.astype(str).to_numpy(),
                      "p": MB.combine_tum(c, df.ADEP_mvt.to_numpy(), yt="yt_tum", cap_non_lirf=3500.0),
                      "mu": c["mu"], "yt": c["yt_tum"], "p24": c["p24"], "d": c["d"]})
    for dosya, cols in (("base_v126_train_v1", ["MVT_HOUR_UTC", "MVT_DAYOFWEEK"]), ("base_v126_rank_v1", ["MVT_HOUR_UTC", "MVT_DAYOFWEEK"]),
                        ("park_b_oz", PB), ("yanki_te_oz", YTE)):
        if all(k in X for k in cols):
            continue
        x = pd.read_parquet(config.DATA_DIR / f"{dosya}.parquet", columns=["MVT_ID_mvt"] + cols)
        x = x[x.MVT_ID_mvt.isin(X.MVT_ID_mvt)].drop_duplicates("MVT_ID_mvt")
        if len(x):
            X = X.merge(x, on="MVT_ID_mvt", how="left")
    return X


def modeller(X, r):
    from catboost import CatBoostRegressor
    return [CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10, random_seed=s, verbose=0,
                              thread_count=8, cat_features=["apt"]).fit(X[F], r) for s in TOHUM]


def tahmin(ms, X):
    return np.mean([m.predict(X[F]) for m in ms], axis=0)


def egitim_verisi():
    df = MB.load_b("train")[["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B]]
    df["r"] = MB.regimes(MB.load_b("train"))
    o = lambda e: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f"BM_cv_tum{e}_oof.parquet"),  # noqa: E731
                                           on="MVT_ID_mvt", how="left")
    P, M = o("_dparkktah"), o("_dparkktahkomgspcyte_cb")
    c = {k: P[f"bilesen_{k}"].to_numpy(float) for k in ("p24", "yt_tum", "d", "d24")}
    c["mu"] = 0.5 * M["bilesen_mu"].to_numpy(float) + 0.5 * M["bilesen_mu_cb"].to_numpy(float)
    X = ozellik(df, c)
    assert (X.MVT_ID_mvt.to_numpy() == df.MVT_ID_mvt.to_numpy()).all()
    return df, X


def cv():
    df, X = egitim_verisi()
    y, p = df[MB.TARGET_B].to_numpy(float), X.p.to_numpy()
    apt8 = df.ADEP_mvt.isin(APT8).to_numpy()
    egit = apt8 & (df.r != 2).to_numpy()
    r_oof = np.zeros(len(df))
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr = egit & ~va_m
        ms = modeller(X[tr], (y - p)[tr])
        te = apt8 & va_m
        r_oof[te] = tahmin(ms, X[te])
        print(f"  {q}: eğitim {tr.sum():,}, sınama {te.sum():,}", flush=True)
    yeni = np.maximum(60.0, p + r_oof)
    sat = {}
    for ad, q in (("melez (v309 B)", p), ("melez + L2", yeni)):
        e2 = (q - y) ** 2
        r = {}
        for kume, aa in (("6 hav.", APT6), ("8 hav.", APT8)):
            a = df.ADEP_mvt.isin(aa).to_numpy()
            r[f"{kume} 24 sa hariç"] = np.sqrt(e2[a & (df.r != 2).to_numpy()].mean())
            r[f"{kume} dahil"] = np.sqrt(e2[a].mean())
        a6 = df.ADEP_mvt.isin(APT6).to_numpy() & (df.r != 2).to_numpy()
        for q_, ms in V.QUARTERS.items():
            r[q_ + " (6)"] = np.sqrt(e2[a6 & df.MVT_MONTH.isin(ms).to_numpy()].mean())
        for a_ in APT8:
            r[a_] = np.sqrt(e2[(df.ADEP_mvt == a_).to_numpy() & (df.r != 2).to_numpy()].mean())
        sat[ad] = r
    print(pd.DataFrame(sat).round(1).to_string())
    pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt, "r": r_oof}).to_parquet(config.DATA_DIR / "preds" / "B_l2_cv.parquet", index=False)


def uretim(ek):
    df, X = egitim_verisi()
    y, p = df[MB.TARGET_B].to_numpy(float), X.p.to_numpy()
    egit = df.ADEP_mvt.isin(APT8).to_numpy() & (df.r != 2).to_numpy()
    ms = modeller(X[egit], (y - p)[egit])
    t = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_B_tum_{ek}.parquet")
    c = {k: t[f"bilesen_{k}"].to_numpy(float) for k in ("mu", "p24", "yt_tum", "d", "d24")}
    Xt = ozellik(t[["MVT_ID_mvt", "ADEP_mvt"]], c)
    assert np.allclose(Xt.p.to_numpy(), t.taksi.to_numpy(float)), "test melezi bileşenlerinden yeniden üretilemedi"
    m = t.ADEP_mvt.isin(APT8).to_numpy()
    r = np.zeros(len(t))
    r[m] = tahmin(ms, Xt[m])
    o = t.copy()
    o["taksi"] = np.maximum(60.0, t.taksi.to_numpy(float) + r)
    o.to_parquet(config.DATA_DIR / "preds" / f"TEST_B_tum_{ek}l2.parquet", index=False)
    print(f"test: {m.sum():,} satır, ekleme ort {r[m].mean():+.1f}, RMS {np.sqrt(np.mean(r[m] ** 2)):.1f}")
    print(pd.DataFrame({"apt": t.ADEP_mvt, "r": r})[m].groupby("apt").r.agg(["size", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cv() if sys.argv[1] == "cv" else uretim(sys.argv[2])

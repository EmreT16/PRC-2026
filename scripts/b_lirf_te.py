"""LIRF Model B rejim sınıflandırıcısına havayolu / uçuş bazlı yer tutucu oranı (hedef kodlaması) eklemek.

Bulgu (30.09, 2025 LIRF): gecikmeli (> 45 dk) kalkışlarda yer tutucu (|BLOCK − SCHED| ≤ 60 sn) oranı havayoluna göre
%0 … %99,6 (ISR %99,6, EXS %34, THY %30, ETH %26; ITY/KLM/SWR ~%0); ilk ve ikinci yarı yıl korelasyonu 0,947; NM'siz
satırlarda havayolu oranının AUC'u 0,872. v126 LIRF sınıflandırıcısı ise yalnızca ~1.500 NM'siz satırdan öğreniyor
(havayolu seyrek kategori). Off-block'un tarifeye eşit kaydı çoğunlukla havayoluna bağlı.
Burada oran TÜM LIRF kalkışlarından (NM'li + NM'siz) hesaplanır; eğitim satırlarına ay dışı (OOF), teste 12 aydan.
Kullanım: python scripts/b_lirf_te.py cv      → çeyrek CV: v126 LIRF ↔ v126 LIRF + oran özellikleri (ön koşul dışı)
          python scripts/b_lirf_te.py uretim  → TEST_B_lirf_te.parquet (v152'nin LIRF B satırları yerine)
"""
import os
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

RAW = config.PRC2_DIR / "data"
PRED = config.DATA_DIR / "preds"
V126 = config.PRC4_DIR / "submissions" / "strong-moose_v126.parquet"
GEC_ESIK, ONCUL = 2700.0, 10.0
ANAHTAR = ("hy", "park")  # 08.10: üretimdeki (v167 → v313) ayar; ("hy", "ucus") 30.09'da Q1'i bozduğu için alınmamıştı  # cv seçeneği: anahtar=hy | ucus | hy,ucus; oncul=<n>


def lirf_ham():
    kol = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
           "SCHED_TIME_UTC_mvt", "STAND_mvt"]
    parca = []
    for m in range(1, 13):
        f = RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                   "training_2025-12-01_2026-01-01.parquet")
        x = pd.read_parquet(f, columns=kol)
        parca.append(x[x["PHASE_mvt"].eq("DEP") & x["ADEP_mvt"].eq("LIRF")].assign(ay=m))
    d = pd.concat(parca, ignore_index=True)
    d["gec"] = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["yt"] = ((d["BLOCK_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs() <= 60).astype(float)
    return d


def anahtarlar(fl, df=None):
    fl = fl.fillna("").astype(str)
    hy = fl.map(lambda s: (re.match(r"^([A-Z]{3}|[A-Z]{2}|[A-Z]\d|\d[A-Z])", s) or [None, "?"])[1] if s else "?")
    out = {"hy": hy, "ucus": fl.where(fl.ne(""), "?")}
    if df is not None:
        out["park"] = df["STAND_mvt"].fillna("?").astype(str) if "STAND_mvt" in df else pd.Series("?", index=df.index)
        out["varis"] = df["ADES_mvt"].fillna("?").astype(str) if "ADES_mvt" in df else pd.Series("?", index=df.index)
    return out


def oran(ref, hedef_anahtar, ref_anahtar, dislanan_ay=None):
    """Gecikmeli referans satırlarında yer tutucu oranı (Bayes yumuşatmalı), anahtara göre."""
    r = ref[ref["gec"] > GEC_ESIK]
    if dislanan_ay is not None:
        r = r[r["ay"] != dislanan_ay]
    p0 = r["yt"].mean()
    g = r.groupby(ref_anahtar.loc[r.index])["yt"].agg(["sum", "size"])
    o = (g["sum"] + ONCUL * p0) / (g["size"] + ONCUL)
    return hedef_anahtar.map(o).fillna(p0).to_numpy(float), hedef_anahtar.map(g["size"]).fillna(0).to_numpy(float)


def te_ozellik(df, ref, oof):
    """df (MVT_ID_mvt, FLIGHT_mvt, MVT_MONTH) için TE sütunları; oof=True ise her satır kendi ayı dışlanarak."""
    ka, kr = anahtarlar(df["FLIGHT_mvt"], df), anahtarlar(ref["FLIGHT_mvt"], ref)
    out = {}
    for ad in ANAHTAR:
        if oof:
            v, n = np.zeros(len(df)), np.zeros(len(df))
            for m in np.unique(df["MVT_MONTH"]):
                i = (df["MVT_MONTH"] == m).to_numpy()
                v[i], n[i] = oran(ref, ka[ad][i], kr[ad], dislanan_ay=m)
        else:
            v, n = oran(ref, ka[ad], kr[ad])
        out[f"YT_ORAN_{ad.upper()}"], out[f"YT_N_{ad.upper()}"] = v, n
    return pd.DataFrame(out, index=df.index)


def b_satirlari(bolum):
    d = MB.load_b(bolum)
    d = d[d["ADEP_mvt"].eq("LIRF")].reset_index(drop=True)
    f = config.RANK_RAW if bolum == "rank" else None
    if f is None:
        ham = lirf_ham()[["MVT_ID_mvt", "FLIGHT_mvt"]]
    else:
        ham = pd.read_parquet(f, columns=["MVT_ID_mvt", "FLIGHT_mvt"])
    return d.merge(ham, on="MVT_ID_mvt", how="left", validate="one_to_one")


TOHUM = int(os.environ.get("PRC_LIRF_TOHUM", "1"))


def fit_lirf(tr, ap, feats, ek_tr=None, ek_ap=None):
    """v126 LIRF bileşenleri (model_b.fit_predict_v126b ile aynı ayar); ek_* verilirse sınıflandırıcıya eklenir."""
    cats = [c for c in MB.V126B_CATS if c in feats]
    X_tr, X_ap = MB._v126b_frame(tr, feats), MB._v126b_frame(ap, feats)
    if ek_tr is not None:
        X_tr = pd.concat([X_tr.reset_index(drop=True), ek_tr.reset_index(drop=True)], axis=1)
        X_ap = pd.concat([X_ap.reset_index(drop=True), ek_ap.reset_index(drop=True)], axis=1)
    y = tr[MB.TARGET_B].to_numpy(float)
    reg = np.zeros(len(y), dtype=int)
    reg[(y >= 2000) & (y < 40000)] = 1
    reg[y >= 40000] = 2
    # 09.10: PRC_LIRF_TOHUM=N → sınıflandırıcı N tohumla (42, 43, …; feature_fraction 0,7 tohuma bağlı), olasılıklar ortalanır.
    # Neden: final ↔ v320 LIRF B farkı RMS 350 (yeniden eğitimde p'nin oynaması; p2 · 87.400 kaldıracı). Varsayılan 1 = eski.
    p = np.mean([lgb.train(dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.03, num_leaves=31,
                                max_depth=5, min_child_samples=6, feature_fraction=0.70, random_state=42 + s, n_jobs=8,
                                verbosity=-1), lgb.Dataset(X_tr, label=reg, categorical_feature=cats),
                           num_boost_round=200).predict(X_ap) for s in range(TOHUM)], axis=0)
    Xr_tr, Xr_ap = MB._v126b_frame(tr, feats), MB._v126b_frame(ap, feats)
    r0 = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.04, num_leaves=15, max_depth=4,
                        min_child_samples=6, verbosity=-1, n_jobs=8),
                   lgb.Dataset(Xr_tr[reg == 0], label=y[reg == 0], categorical_feature=cats), num_boost_round=100)
    d = ap[MB.DELAY].to_numpy(float)
    parts = (r0.predict(Xr_ap), np.clip(d, 60.0, None), np.where(d > 40000, d, 87400.0))
    pred = np.clip(p[:, 0] * parts[0] + p[:, 1] * parts[1] + p[:, 2] * parts[2], 60.0, None)
    return pred, p, parts


def cv():
    sys.stdout.reconfigure(encoding="utf-8")
    ref = lirf_ham()
    df = b_satirlari("train")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    y = df[MB.TARGET_B].to_numpy(float)
    pre = MB.on_kosul_24s(df)
    tahmin = {"v126": np.full(len(df), np.nan), "v126+oran": np.full(len(df), np.nan)}
    ll = {k: [] for k in tahmin}
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr, va = df[~va_m].reset_index(drop=True), df[va_m].reset_index(drop=True)
        # oran: eğitim satırlarına kendi ayları dışlanarak; doğrulamaya doğrulama çeyreği dışlanarak (ref'ten çıkarılır)
        ref_q = ref[~ref["ay"].isin(V.QUARTERS[q])]
        ek_tr = te_ozellik(tr, ref_q, oof=True)
        ek_va = te_ozellik(va, ref_q, oof=False)
        for ad, (a, b) in (("v126", (None, None)), ("v126+oran", (ek_tr, ek_va))):
            pred, p, _ = fit_lirf(tr, va, feats, a, b)
            tahmin[ad][va_m] = pred
            yv = va[MB.TARGET_B].to_numpy(float)
            reg = np.where(yv >= 40000, 2, np.where(yv >= 2000, 1, 0))
            ll[ad].append(-np.mean(np.log(np.clip(p[np.arange(len(reg)), reg], 1e-6, 1))))
    rows = {}
    for grp, m in (("LIRF ön koşul dışı", ~pre), ("LIRF hepsi", np.ones(len(df), bool))):
        for ad, p in tahmin.items():
            rows[(grp, ad)] = {"satır": int(m.sum()), "RMSE": float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))}
    print(pd.DataFrame(rows).T.round(1).to_string())
    for ad in ll:
        print(f"{ad}: çeyrek logloss {np.round(ll[ad], 4).tolist()} | ort {np.mean(ll[ad]):.4f}")
    for qn, ms in V.QUARTERS.items():
        m = df["MVT_MONTH"].isin(ms).to_numpy() & ~pre
        print(f"  {qn} ön koşul dışı: " + " | ".join(f"{ad} {np.sqrt(np.mean((p[m] - y[m]) ** 2)):.1f}"
                                                     for ad, p in tahmin.items()))


def uretim():
    """12 aylık LIRF B satırlarıyla (oran: tüm LIRF kalkışlarından, eğitimde ay dışı) sınıflandırıcı; test LIRF ön koşul
    dışı satırlarına v152'nin formülüyle: p0·μ_tüm + p1·max(gecikme, 60) + p2·(gecikme > 40.000 ? gecikme : 87.400).
    Ön koşullu satırlar değişmez. Çıktı: TEST_B_lirf_te.parquet; ayrıca TEST_B_tum6M_adsb_lirfte.parquet
    (TEST_B_tum6M_adsb'nin LIRF ön koşul dışı satırları yenileriyle)."""
    sys.stdout.reconfigure(encoding="utf-8")
    ref = lirf_ham()
    tr, ap = b_satirlari("train"), b_satirlari("rank")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    ek_tr, ek_ap = te_ozellik(tr, ref, oof=True), te_ozellik(ap, ref, oof=False)
    _, p, parts = fit_lirf(tr, ap, feats, ek_tr, ek_ap)
    # 08.10: PRC_LIRF_MU=<ad> → μ = 0,5 · bilesen_mu + 0,5 · bilesen_mu_cb, TEST_B_<ad>.parquet'ten (hepsi=1 üretimi; CV
    # scripts/b_lirf_mu_cv.py: ön koşul dışı 2.906,1 → 2.885,9); çıktılar PRC_LIRF_EK ekiyle. Yoksa eski μ (TEST_B_tum_hepsi).
    kaynak = os.environ.get("PRC_LIRF_MU")
    if kaynak:
        m_ = pd.read_parquet(PRED / f"TEST_B_{kaynak}.parquet")
        m_ = m_.assign(mu=0.5 * m_.bilesen_mu + 0.5 * m_.bilesen_mu_cb)[["MVT_ID_mvt", "mu"]]
    else:
        m_ = pd.read_parquet(PRED / "TEST_B_tum_hepsi.parquet")[["MVT_ID_mvt", "bilesen_mu"]].rename(columns={"bilesen_mu": "mu"})
    mu = ap[["MVT_ID_mvt"]].merge(m_, on="MVT_ID_mvt", how="left", validate="one_to_one")["mu"].to_numpy(float)
    assert np.isfinite(mu).all(), "LIRF μ eksik"
    ek = os.environ.get("PRC_LIRF_EK", "")
    yeni = np.clip(p[:, 0] * mu + p[:, 1] * parts[1] + p[:, 2] * parts[2], 60.0, None)
    pre = np.asarray(MB.on_kosul_24s(ap), bool)
    eski = pd.read_parquet(PRED / "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "taksi"])
    o = ap[["MVT_ID_mvt"]].merge(eski, on="MVT_ID_mvt", how="left", validate="one_to_one")["taksi"].to_numpy(float)
    out = pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"], "ADEP_mvt": "LIRF", "taksi": np.where(pre, o, yeni), "eski": o,
                        "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], "on_kosul": pre,
                        "YT_ORAN_HY": ek_ap["YT_ORAN_HY"].to_numpy()})
    out.to_parquet(PRED / f"TEST_B_lirf_te{ek}.parquet", index=False)
    d = out["taksi"] - out["eski"]
    print(f"LIRF test B {len(out)} | ön koşul dışı {int((~pre).sum())} | değişen {int((d.abs() > 1e-6).sum())} | "
          f"RMS fark {np.sqrt(np.mean(d[~pre] ** 2)):.0f} | ort fark {d[~pre].mean():.0f}")
    print("fark yüzdelikleri:", np.percentile(d[~pre], [1, 5, 25, 50, 75, 95, 99]).round(0).tolist())
    tam = eski.merge(out[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"}), on="MVT_ID_mvt", how="left")
    tam["taksi"] = tam["yeni"].fillna(tam["taksi"])
    tam[["MVT_ID_mvt", "taksi"]].to_parquet(PRED / f"TEST_B_tum6M_adsb_lirfte{ek}.parquet", index=False)
    print(f"TEST_B_tum6M_adsb_lirfte: {len(tam)} B satırı; LIRF'te değişen {int((d.abs() > 1e-6).sum())}")


if __name__ == "__main__":
    _opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    ANAHTAR = tuple(_opt.get("anahtar", "hy,park").split(","))
    ONCUL = float(_opt.get("oncul", ONCUL))
    {"cv": cv, "uretim": uretim}[sys.argv[1]]()

"""G2 — gecikmeli uçuşlarda L2 ikinci aşama (07.10 gece; hata anatomisi).

Neden: gecikmeli uçuşlarda kayıt çoğu kez planlı saati (tarife / LOBT / EOBT) kopyalıyor → gerçek değer iki rejimin karışımı.
RMSE'nin en iyisi koşullu ortalama, ama Model A bileşenleri ve düzeltici sağlam kayıplarla (Huber δ 400 / 600) eğitildi
→ bu satırlarda sistematik eksik tahmin (CV: tarife kopyalarında gerçek ort. 2.666, tahmin 2.017).
Ne: Model A + düzelticinin kalıntısını (y − p) yalnız "kopya olsaydı kalıntı > 15 dk" satırlarında (min(MVT − tarife,
MVT − LOBT, MVT − EOBT) − p > 900; etiketsiz) küçük bir L2 LGB ile öğren, ekle.
Seçim yalnız CV. Oca ↔ Tem (bir ayda öğren öbüründe sına): 16 ayar × 3 tohumun hepsinde iki yönde de iyi (SSE −%0,06 …
−%0,19); seçilen en kararlı ayar (7 yaprak, yaprakta ≥ 1.000, lr 0,01, 300 tur) −%0,19. AMA Oca + Tem ile öğrenilen G2
Mart'ta +%0,10, Ekim'de 0,00 → yalnız iki ayla öğrenmek kırılgan. Düzeltici OOF'u olan 5 ayla (Oca, Mar, Tem, Eki, Kas)
ay dışarıda bırakmalı sınama: Oca −%0,14, Mar +%0,03, Tem −%0,26, Eki −%0,05, Kas −%0,02, toplam −%0,115 → üretim 5 ayla.
Üretim: 5 ayın tamamıyla 3 tohum eğitilir, TEST_D_<aday>.parquet'in taksi sütununa uygulanır (60 sn alt sınır)
→ TEST_D_<aday>_g2.parquet. Kalibrasyon (g_kalibre) bundan SONRA uygulanır (CV'deki p kalibrasyon öncesi).
Kullanım: python scripts/gecikme_asama2.py <aday, ör. D_kqgpPe_adsb_40o> [ha=1]
  ha=1: eğitim tabanı H1 (havalimanı ağırlıkları); test adayı da H1'li olmalı (ör. D_kqgpPe_adsb_40o_ha)
Ortam: PRC_HARMAN=kqgpPe PRC_ADSB_OZ=adsb5 (CV tabanı final Model A ile aynı).
"""
import glob
import sys
from pathlib import Path

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

P = dict(objective="l2", learning_rate=0.01, num_leaves=7, min_data_in_leaf=1000, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1)
TUR = 300
# 08.10: PRC_G2_LGB="num_leaves=31,min_data_in_leaf=200,learning_rate=0.03" PRC_G2_TUR=600 → LightGBM kapasite sınaması
# (CatBoost G2'nin kazancı kapasiteden mi geliyor); çıktı _g2k
if os.environ.get("PRC_G2_LGB"):
    P.update({k: (float(v) if "." in v else int(v)) for k, v in (x.split("=") for x in os.environ["PRC_G2_LGB"].split(","))})
TUR = int(os.environ.get("PRC_G2_TUR", TUR))
TOHUM = (1, 2, 3)
ESIK = float(os.environ.get("PRC_G2_ESIK", 900))  # 08.10: eşik ortamdan değiştirilebilir (çıktı _g2e<eşik>)
PLAN = ["SCHED", "LOBT", "IOBT", "EOBT", "AOBT"]
F = ["p", "taban", "rhat", "NM_TAXI_DIFF_SEC"] + [f"G_{c}" for c in PLAN] + ["G_MIN"] + [f"R_{c}" for c in PLAN] + \
    ["ADSB_ILK", "ADSB_HAREKET", "OP_YER", "OP_PARK_SON", "MVT_HOUR_UTC", "apt"]
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
# PRC_G2_PARKB=1 (07.10 gece): park yeri dizisi ve komşu tahmini taksi (scripts/park_b_ozellik.py) G2'ye de; çıktı _g2pb
import os  # noqa: E402
G2_PB = os.environ.get("PRC_G2_PARKB") == "1"
PB = ["PB_ON1", "PB_ON1_HY", "PB_ON1_TIP", "PB_AYNI", "PB_SON1", "PD_ON1", "PD_SON1", "PB_KT15", "PB_KT15_P"]
if G2_PB:
    F = F + PB + ["dPB_ON1", "dPB_AYNI", "dPB_KT15"]
# PRC_G2_MODEL=cb | lgb+cb (08.10): ikinci model ailesi (CatBoost, derinlik 4, 600 yineleme, aynı 3 tohum); çıktı _g2<model>
G2_MODEL = os.environ.get("PRC_G2_MODEL", "lgb")
# PRC_G2_TERS=1 (08.10): G2b — aynı ikinci aşama G2'nin DIŞINDAKİ satırlarda (G_MIN − p ≤ eşik); kümeler ayrık → kazançlar
# toplanır. Test tarafında G2'den önceki aday (ör. _ha) verilir; G2 ile birleştirme ayrı adımda. Çıktı _g2t…
G2_TERS = os.environ.get("PRC_G2_TERS") == "1"
# PRC_G2_BIL=1 (08.10): beş taban bileşenin ayrı taksi tahminleri (p'ye göre fark) ve uyuşmazlıkları; "akıllı harman".
# Eğitim: bileşenlerin OOF dosyaları (A.BILESEN); test: TEST_<A.TEST_AD> sütunları. Çıktı _g2…bil
G2_BIL = os.environ.get("PRC_G2_BIL") == "1"
# PRC_G2_ADSB=1 (08.10): düzelticinin ADS-B özelliklerinin tamamı (yalnız ILK / HAREKET yerine) + p'ye göre farkları; CatBoost
# L2 aşaması Huber düzelticinin tam kullanamadığı ADS-B sinyalini yakalayabilir. Çıktı _g2…adsb
G2_ADSB = os.environ.get("PRC_G2_ADSB") == "1"
ADSB_EK = ["ADSB_ESLESME", "ADSB_TAKSI_BAS", "ADSB_PARK_L", "ADSB_PARK_U", "ADSB_PARK_ARALIK", "ADSB_PARK_ILK", "ADSB_TO_FARK",
           "ADSB_N", "ADSB_BOSLUK", "ADSB_BOSLUK5", "ADSB_ILK_GS", "ADSB_ILK_DURMA", "ADSB_ILK_UZAK", "ADSB_YOL", "ADSB_DURMA",
           "ADSB_KOSU"]
ADSB_FARK = ["ADSB_ILK", "ADSB_HAREKET", "ADSB_TAKSI_BAS", "ADSB_PARK_U"]
if G2_ADSB:
    F = F + ADSB_EK + [f"d{c}" for c in ADSB_FARK]


def _adsb_fark(d):
    for c in ADSB_FARK:
        d[f"d{c}"] = d[c] - d["p"]
    return d
BIL_AD = ["xgb_P", "gcb_P4", "e1h3_P6", "lgbh3_P6", "dw_P5"]
if G2_BIL:
    F = F + [f"dBIL_{a}" for a in BIL_AD] + ["BIL_STD"]


def _bil_fark(d):
    for a in BIL_AD:
        d[f"dBIL_{a}"] = d[f"BIL_{a}"] - d["p"]
    return d
# PRC_G2_YTE=1 (08.10, LIRF hata anatomisi: tarife kopyaları LIRF SSE'sinin %47'si): yankı geçmişi (kaydın tarifeyi kopyalama
# oranı; uçuş / havayolu / varış / park, ay dışı hedef kodlama, data/yanki_te_oz.parquet) + beklenen kopya kalıntısı
# (oran × (MVT − tarife − p)); çıktı _g2yte
G2_YTE = os.environ.get("PRC_G2_YTE") == "1"
YTE = ["YTE_UCUS", "YTE_HY", "YTE_VARIS", "YTE_PARK", "YTE_N_UCUS"]
if G2_YTE:
    F = F + YTE + ["BEK_UCUS", "BEK_HY"]


def _bek(d):
    for c, k in (("BEK_UCUS", "YTE_UCUS"), ("BEK_HY", "YTE_HY")):
        d[c] = d[k] * d["R_SCHED"]
    return d


def ekle(d, dep, base, adsb_desen):
    k = ["MVT_ID_mvt"] + [f"{c}_DIFF_SEC" for c in PLAN]
    d = d.drop(columns=[c for c in k[1:] + ["MVT_HOUR_UTC"] if c in d])
    d = d.merge(pd.read_parquet(dep, columns=k), on="MVT_ID_mvt", how="left")
    d = d.merge(pd.read_parquet(base, columns=["MVT_ID_mvt", "MVT_HOUR_UTC"]), on="MVT_ID_mvt", how="left")
    a = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "ADSB_ILK", "ADSB_HAREKET"] + (ADSB_EK if G2_ADSB else []))
                   for f in glob.glob(adsb_desen)])
    d = d.merge(a.drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")
    o = pd.read_parquet(config.DATA_DIR / "opdi_oz.parquet", columns=["MVT_ID_mvt", "OP_YER", "OP_PARK_SON"])
    d = d.merge(o, on="MVT_ID_mvt", how="left")
    for c in PLAN:
        d[f"G_{c}"] = -d[f"{c}_DIFF_SEC"]  # MVT − X (X_DIFF = X − MVT, PRC_2/src/features.py:127-131)
        d[f"R_{c}"] = d[f"G_{c}"] - d["p"]
    d["G_MIN"] = d[["G_SCHED", "G_LOBT", "G_EOBT"]].min(axis=1)
    if G2_PB:
        d = d.merge(pd.read_parquet(config.DATA_DIR / "park_b_oz.parquet", columns=["MVT_ID_mvt"] + PB), on="MVT_ID_mvt", how="left")
        for c in ("PB_ON1", "PB_AYNI", "PB_KT15"):
            d["d" + c] = d[c] - d["p"]
    if G2_BIL:
        d = _bil_fark(d)
    if G2_ADSB:
        d = _adsb_fark(d)
    if G2_YTE:
        d = _bek(d.merge(pd.read_parquet(config.DATA_DIR / "yanki_te_oz.parquet", columns=["MVT_ID_mvt"] + YTE), on="MVT_ID_mvt", how="left"))
    d["apt"] = pd.Categorical(d["ADEP_mvt"].astype(str), categories=APT)
    d["gec"] = ((d["G_MIN"] - d["p"]) > ESIK) != G2_TERS  # G2_TERS: tümleyen (NaN G_MIN dahil)
    return d


# (Model A OOF bölmesi, aylar, düzelticinin aynı ayarlı CV kalıntısı: opdi=1 opdi_satir=1, sinama ayları farklı)
import os  # noqa: E402
# PRC_DUZ_CV=parkb (07.10 gece): düzeltici kalıntıları park yeri özellikli düzelticinin (_42o) CV'lerinden
_CV = {"": ("ADSB_duz_cv_opdi_satir", "ADSB_duz_cv_bosluk_taban", "ADSB_duz_cv_opdi_satir_kas"),
       "parkb": ("ADSB_duz_cv_parkb", "ADSB_duz_cv_bosluk_parkb", "ADSB_duz_cv_parkb_kas"),
       # 08.10: düzeltici LightGBM + CatBoost (model=lgb+cb, parkb=1; scripts/zincir_duz_lgbcb.sh)
       "lgbcb": ("ADSB_duz_cv_lgbcb", "ADSB_duz_cv_bosluk_lgbcb", "ADSB_duz_cv_lgbcb_kas"),
       # 08.10: final hattı parkb OOF'larını _f ekiyle yeniden üretir (yeniden üretilebilirlik)
       "parkbf": ("ADSB_duz_cv_parkb_f", "ADSB_duz_cv_bosluk_parkb_f", "ADSB_duz_cv_parkb_kas_f"),
       # 09.10 akşam: parkbsh = parkb + Şubat / Haziran 2025 (düzeltici CV sinama=2,6; scripts/zincir_subhaz_cv.sh) — yalnız sınama
       "parkbsh": ("ADSB_duz_cv_parkb", "ADSB_duz_cv_bosluk_parkb", "ADSB_duz_cv_parkb_kas")}[os.environ.get("PRC_DUZ_CV", "")]
KAYNAK = (("split_1", (1, 7), _CV[0]), ("split_4", (3,), _CV[1]), ("split_5", (10,), _CV[1]), ("split_4", (11,), _CV[2]))
if os.environ.get("PRC_DUZ_CV") == "parkbsh":
    KAYNAK = KAYNAK + (("split_2", (2,), "ADSB_duz_cv_parkb_subhaz"), ("split_3", (6,), "ADSB_duz_cv_parkb_subhaz"))


def h1_taban(d):
    """ha=1 (07.10 gece): eğitim p'si H1 (havalimanı ağırlıkları, scripts/havalimani_agirlik.py) tabanıyla, ay dışarıda
    bırakmalı seçilmiş ağırlıklarla (CV'deki gibi). H1 + G2 birlikte ay dışarıda: SSE −%0,258 (H1 −%0,129, G2 −%0,115)."""
    import json
    import havalimani_agirlik as H
    w = json.loads(A.AGIRLIK.read_text(encoding="utf-8"))
    ad, W0 = w["bilesenler"], np.array(w["agirlik"])
    h = H.veri(ad)
    NM, r, ay, O = h.NM_TAXI_DIFF_SEC.to_numpy(float), h.rhat.to_numpy(float), h.MVT_MONTH.to_numpy(), h[ad].to_numpy(float)
    pH = np.empty(len(h))
    for m in sorted(np.unique(ay)):
        W, te = H.agirliklar(h, ad, W0, ay != m), ay == m
        pH[te] = np.maximum(60.0, np.maximum(60.0, NM[te] - H.uygula(h[te], ad, W, O[te] @ W0)) + r[te])
    d = d.merge(pd.DataFrame({"MVT_ID_mvt": h.MVT_ID_mvt.to_numpy(), "pH": pH}), on="MVT_ID_mvt", how="left", validate="one_to_one")
    d["p"] = d["pH"].fillna(d["p"])
    for c in PLAN:
        d[f"R_{c}"] = d[f"G_{c}"] - d["p"]
    if G2_PB:
        for c in ("PB_ON1", "PB_AYNI", "PB_KT15"):
            d["d" + c] = d[c] - d["p"]
    if G2_YTE:
        d = _bek(d)
    if G2_BIL:
        d = _bil_fark(d)
    if G2_ADSB:
        d = _adsb_fark(d)
    d["gec"] = ((d["G_MIN"] - d["p"]) > ESIK) != G2_TERS  # G2_TERS: tümleyen (NaN G_MIN dahil)
    return d.drop(columns="pH")


def egitim():
    parca = []
    for sp, aylar, rf in KAYNAK:
        d = A.d_dw_split(sp)
        d = d[d.MVT_MONTH.isin(aylar)].copy()
        if G2_BIL:
            for a in BIL_AD:
                f, col = A.BILESEN[a]
                v = d[["MVT_ID_mvt"]].merge(pd.read_parquet(A.PRED / f"{f}_{sp}.parquet", columns=["MVT_ID_mvt", col]),
                                            on="MVT_ID_mvt", how="left", validate="one_to_one")[col].to_numpy(float)
                d[f"BIL_{a}"] = np.maximum(60.0, d.NM_TAXI_DIFF_SEC.to_numpy(float) - v)
            d["BIL_STD"] = d[[f"BIL_{a}" for a in BIL_AD]].std(axis=1)
        d["taban"] = A.taksi(d.NM_TAXI_DIFF_SEC, d.ofset_taban)
        d["yt"] = d.NM_TAXI_DIFF_SEC - d.y
        d = d.merge(pd.read_parquet(config.DATA_DIR / "preds" / f"{rf}.parquet"), on="MVT_ID_mvt", how="left")
        d["rhat"] = d["rhat"].fillna(0.0)
        d["p"] = np.maximum(60.0, d.taban + d.rhat)
        parca.append(d)
    return ekle(pd.concat(parca, ignore_index=True), config.DEP_TRAIN, config.DATA_DIR / "base_v126_train_v1.parquet",
                str(A.ADSB_DIR / "oz_2025-*.parquet"))


def _cbx(X):
    X = X.copy()
    X["apt"] = X["apt"].astype(str)
    return X


def modeller(d, maske):
    X, r = d.loc[maske, F], (d.yt - d.p)[maske].to_numpy()
    ms = [lgb.train(dict(P, seed=s), lgb.Dataset(X, r), TUR) for s in TOHUM] if "lgb" in G2_MODEL else []
    if "cb" in G2_MODEL:
        from catboost import CatBoostRegressor
        ms += [CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10, random_seed=s, verbose=0,
                                 thread_count=8, cat_features=["apt"]).fit(_cbx(X), r) for s in TOHUM]
    return ms


def tahmin(ms, X):
    return np.mean([m.predict(X) if isinstance(m, lgb.Booster) else m.predict(_cbx(X)) for m in ms], axis=0)


def main(aday, ha=False):
    d = egitim()
    if ha:
        d = h1_taban(d)
    y, p = d.yt.to_numpy(), d.p.to_numpy()
    S0 = np.sum((p - y) ** 2)
    print(f"CV satırı {len(d):,} (RMSE {np.sqrt(S0 / len(d)):.2f}); G2 satırı {int(d.gec.sum()):,}, SSE payı %{100 * np.sum(((p - y) ** 2)[d.gec]) / S0:.1f}")
    top = 0.0
    r_oof = np.zeros(len(d))
    for ay in sorted(d.MVT_MONTH.unique()):
        te = (d.MVT_MONTH == ay).to_numpy()
        m = te & d.gec.to_numpy()
        r = np.zeros(len(d))
        r[m] = tahmin(modeller(d, d.gec & (d.MVT_MONTH != ay)), d.loc[m, F])
        e0, e1 = np.sum((p[te] - y[te]) ** 2), np.sum((np.maximum(60, p[te] + r[te]) - y[te]) ** 2)
        print(f"  ay {ay:2d} dışarıda: RMSE {np.sqrt(e0 / te.sum()):.2f} → {np.sqrt(e1 / te.sum()):.2f} (SSE %{100 * (e1 / e0 - 1):+.2f})")
        top += e1 - e0
        r_oof[te] = r[te]
    print(f"  toplam ΔSSE %{100 * top / S0:+.3f}")
    # 08.10: OOF kaydı (Y1 kalibrasyonu benzetimi için: havalimanı-ay ortalaması çıkarılınca kalan kazanç)
    ek = os.environ.get("PRC_G2_OOF")
    if ek:
        pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.to_numpy(), "apt": d.ADEP_mvt.astype(str).to_numpy(), "ay": d.MVT_MONTH.to_numpy(),
                      "y": y, "p": p, "r": r_oof}).to_parquet(config.DATA_DIR / "preds" / f"G2_cv_{ek}.parquet", index=False)
    ms = modeller(d, d.gec)
    t = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_{aday}.parquet")
    t["p"], t["taban"] = t["taksi"].astype(float), t["taksi_taban"].astype(float)
    if G2_BIL:
        b = t[["MVT_ID_mvt"]].merge(pd.read_parquet(A.PRED / f"TEST_{A.TEST_AD}.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC"] + BIL_AD),
                                    on="MVT_ID_mvt", how="left", validate="one_to_one")
        for a in BIL_AD:
            t[f"BIL_{a}"] = np.maximum(60.0, b.NM_TAXI_DIFF_SEC.to_numpy(float) - b[a].to_numpy(float))
        t["BIL_STD"] = t[[f"BIL_{a}" for a in BIL_AD]].std(axis=1)
    t = ekle(t, config.DEP_RANK, config.DATA_DIR / "base_v126_rank_v1.parquet", str(A.ADSB_DIR / "oz_2026-*.parquet"))
    r = np.zeros(len(t))
    r[t.gec.to_numpy()] = tahmin(ms, t.loc[t.gec, F])
    yeni = np.maximum(60.0, t["p"].to_numpy() + r)
    print(f"test: G2 satırı {int(t.gec.sum()):,} / {len(t):,}; ekleme ort {r[t.gec].mean():+.1f} sn, RMS {np.sqrt(np.mean(r[t.gec] ** 2)):.1f}, "
          f"çeyrekler {np.percentile(r[t.gec], [5, 25, 50, 75, 95]).round(0).tolist()}")
    for apt, g in t[t.gec].assign(r=r[t.gec.to_numpy()]).groupby("ADEP_mvt", observed=True):
        print(f"  {apt}: {len(g):,} satır, ort {g.r.mean():+.1f}")
    eski = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_{aday}.parquet")
    assert np.array_equal(eski.MVT_ID_mvt.to_numpy(), t.MVT_ID_mvt.to_numpy())
    eski["taksi"] = yeni
    cik = config.DATA_DIR / "preds" / f"TEST_{aday}_g2{'t' if G2_TERS else ''}{'pb' if G2_PB else ''}{'yte' if G2_YTE else ''}{'bil' if G2_BIL else ''}{'adsb' if G2_ADSB else ''}{'' if G2_MODEL == 'lgb' else G2_MODEL.replace('+', '')}{'' if ESIK == 900 else f'e{ESIK:.0f}'}{'k' if os.environ.get('PRC_G2_LGB') else ''}.parquet"
    eski.to_parquet(cik, index=False)
    print(f"yazıldı: {cik}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(x.split("=", 1) for x in sys.argv[2:] if "=" in x)
    main(sys.argv[1], opt.get("ha") == "1")

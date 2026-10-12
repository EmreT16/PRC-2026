"""ADS-B düzeltici v5 — Model B (NM'siz) satırları (30.09). adsb_b_duzeltici.py'nin (v1 özellikleri, 2 ay) yerine.

Model A satırlarıyla eğitilen v5 düzeltici (6 ay, oz=4 park/kayıt alışkanlığı, uçak tipi, 255 yaprak / 800 tur) NM'siz
satırlara uygulanır (nm, taban_std = NaN). Yerel (±30 dk komşu) ve merkez özellikleri, NM'siz satırlar için Model A
satırlarıyla birlikte hesaplanır (yalnızca NM'siz satırlar arasında komşu azdır).
Taban: gönderimdeki Model B tahmini (EHAM/LSZH/EDDF v126_B, diğerleri tum6M); CV'de karşılıkları BM_cv_oof / BM_cv_tum_oof.
Kullanım: python scripts/adsb_b_duzeltici5.py cv
          python scripts/adsb_b_duzeltici5.py uretim apt=EHAM,EDDF,... [k=1] [ad=tum6M_adsb5_lirfte]
            → data/preds/TEST_B_<ad>.parquet (tüm B satırları; g_gonderim.py b=<ad>)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_duzeltici as D  # noqa: E402
from src import adsb as A, config, model_b as MB  # noqa: E402

PRED = config.DATA_DIR / "preds"
V126_APT = ["EHAM", "LSZH", "EDDF"]
B_OOF = ""  # boof=<ek> (07.10 gece): CV tabanı BM_cv_tum<ek>_oof.parquet (ör. _dparkktah: park yeri / komşu taksi özellikli B)
AYLAR = (1, 2, 6, 7, 8, 12)
PARAM = dict(D.PARAM, objective="huber", alpha=600.0, num_leaves=255)
MODEL = "lgb"  # 08.10: model=lgb+cb → LightGBM + CatBoost (adsb_duzeltici.egit)


def ayarla(opt=None):
    D.YENI_OZ, D.PARK_OZ, D.TIP_OZ = True, True, True
    D.TUR = 800
    # 05.10 gece (denetim T2): Model A düzelticisine sonradan eklenenler B5'te de açılabilir; varsayılan eski davranış.
    opt = opt or {}
    D.IZ5 = opt.get("iz5") == "1"
    D.KUYRUK_OZ = opt.get("kuyruk") == "1"
    D.NMSIZ = opt.get("nmsiz") == "1"  # 06.10: B satırlarında NaN olan NM girdileri olmadan
    # 07.10: OPDI yer olayları; B çerçevelerine de eklenir (ortak_matris kesişimi)
    D.OPDI_OZ = opt.get("opdi") == "1"
    D.PARKB_OZ = opt.get("parkb") == "1"  # 07.10 gece: park yeri dizisi + komşu taksi (Model A düzelticisinde 196,58 → 196,14)
    D.OPDI_SATIR = opt.get("opdi_satir") == "1"
    D.PLANLI_OZ = opt.get("planli") == "1"  # 10.10: planlı saatler (B satırlarında yalnız tarife dolu; nmsiz=1 NM'ye bağlıları atar)
    assert not D.OPDI_SATIR or D.OPDI_OZ, "opdi_satir=1, opdi=1 ile kullanılır"
    global B_OOF, V126_APT, AYLAR
    B_OOF = opt.get("boof", "")
    if opt.get("aylar") == "12":  # 10.10 gece: CV'de 12 ayın B satırları (değerlendirme kitlesi ~2 kat; önceden 6 ay)
        AYLAR = tuple(range(1, 13))
    if "v126apt" in opt:  # v126apt=EDDF: CV'de tabanı v126_B kalan havalimanları (varsayılan EHAM, LSZH, EDDF)
        V126_APT = [a for a in opt["v126apt"].split(",") if a]
    assert not D.IZ5 or "adsb5" in str(A.ADSB_DIR), "iz5=1, PRC_ADSB_OZ=adsb5 ile kullanılır"


def a_veri():
    d = D.veri(True)
    d, _ = D.damga_ekle(d)
    return d


def damga_ref(a):
    iyi = ((a["ADSB_ESLESME"].fillna(0) > 0) & (a["ADSB_ILK_GS"].fillna(99) < 3) & a["ADSB_HAREKET"].notna()
           & (a["y_taksi"] < 7200))
    return a[iyi].assign(u=a.loc[iyi, "y_taksi"] - a.loc[iyi, "ADSB_HAREKET"])


def damga_ref5(a):
    """07.10: park ayraçlı kayıt alışkanlığı referansı (adsb_duzeltici.damga_ekle'deki ref5 ile aynı tanım)."""
    iyi5 = ((a["ADSB_ESLESME"].fillna(0) > 0) & a["ADSB_PARK_U"].notna() & (a["ADSB_PARK_ARALIK"] <= 120)
            & (a["y_taksi"] < 7200))
    return a[iyi5].assign(u=a.loc[iyi5, "y_taksi"] - a.loc[iyi5, "ADSB_PARK_U"])


def b_hazirla(b, ref, oof, ref5=None):
    """B satırlarına ADS-B, kalkış anı ve kayıt alışkanlığı özellikleri. ref5 verilirse (damga5=1) DAMGA5_* de: önceden B
    satırlarında hesaplanmadığı için ortak_matris'te A satırlarından da düşüyor ve B5'te %100 boş kalıyordu."""
    b = b.merge(A.ozellikler(), on="MVT_ID_mvt", how="left", validate="one_to_one")
    b = b.merge(A.mvt_epoch(b["MVT_ID_mvt"]), on="MVT_ID_mvt", how="left", validate="one_to_one")
    b = D.damga_hazirla(b.drop(columns=[c for c in ("AIRLINE_LETTERS", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt")
                                        if c in b]))
    b = pd.concat([b, D.damga_te(b, ref, oof=oof)], axis=1)
    if ref5 is not None:
        b = pd.concat([b, D.damga_te(b, ref5, oof=oof, onek="DAMGA5")], axis=1)
    if D.OPDI_OZ:
        b = D.opdi_ekle(b)
    if D.PARKB_OZ:
        b = D.parkb_ekle(b)
    if D.PLANLI_OZ:
        b = D.planli_ekle(b)
    b["NM_TAXI_DIFF_SEC"] = np.nan
    b["taban_std"] = np.nan
    return b


def b_egitim_veri(ref, ref5=None):
    tr = MB.load_b("train")
    tr = tr[tr["MVT_MONTH"].isin(AYLAR)][["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", MB.TARGET_B]].reset_index(drop=True)
    oof = pd.read_parquet(PRED / "BM_cv_oof.parquet", columns=["MVT_ID_mvt", "v126_B"])
    tum = pd.read_parquet(PRED / f"BM_cv_tum{B_OOF}_oof.parquet", columns=["MVT_ID_mvt", "tum_mu_yt_tavan3500"])
    b = tr.merge(oof, on="MVT_ID_mvt", how="left").merge(tum, on="MVT_ID_mvt", how="left")
    b["taban"] = np.where(b["ADEP_mvt"].isin(V126_APT), b["v126_B"], b["tum_mu_yt_tavan3500"])
    b["y_taksi"] = b[MB.TARGET_B].astype(float)
    b = b_hazirla(b, ref, oof=True, ref5=ref5)
    return b[b["taban"].notna()].reset_index(drop=True)


def ortak_matris(a, b):
    """A ve B satırlarını birlikte matrise çevirir (yerel/merkez ortak); (Xa, Xb) döner, sütunlar aynı."""
    kol = sorted(set(a.columns) & set(b.columns))
    c = pd.concat([a[kol], b[kol]], ignore_index=True)
    X = D.matris(c, havalimani=False)
    return X.iloc[:len(a)].reset_index(drop=True), X.iloc[len(a):].reset_index(drop=True)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def cv(opt):
    ayarla(opt)
    a = a_veri()
    ref = damga_ref(a)
    ref5 = damga_ref5(a) if opt.get("damga5") == "1" else None
    b = b_egitim_veri(ref, ref5)
    D.TIP_KAT = None
    Xa, Xb = ortak_matris(a, b)
    if "tip" in Xa:
        D.TIP_KAT = Xa["tip"].cat.categories
    ra = (a["y_taksi"] - a["taban"]).to_numpy(float)
    es_a, es_b = D.satirlar(a), D.satirlar(b)
    ay_a, ay_b = a["MVT_MONTH"].to_numpy(), b["MVT_MONTH"].to_numpy()
    rhat = np.zeros(len(b))
    for m in AYLAR:
        tr, te = es_a & (ay_a != m), es_b & (ay_b == m)
        if te.sum() == 0:
            continue
        ms = D.egit(Xa[tr], ra[tr], PARAM, MODEL)
        rhat[te] = D.tahmin(ms, Xb[te])
        print(f"ay {m}: A eğitim {tr.sum():,} → B sınama {te.sum():,}", flush=True)
    y, base = b["y_taksi"].to_numpy(float), b["taban"].to_numpy(float)
    rows = {}
    for k in (0.0, 0.5, 1.0, 1.25, 1.5):  # 07.10: 1,25 / 1,5 da (B'de düzeltme kısılıyor mu; karar CV'ye göre, LB'ye göre değil)
        p = np.maximum(60.0, base + k * rhat)
        row = {f"ay {m}": rmse(p[ay_b == m], y[ay_b == m]) for m in AYLAR}
        row["izli B"] = rmse(p[es_b], y[es_b])
        for apt in sorted(b["ADEP_mvt"].unique()):
            i = (b["ADEP_mvt"] == apt).to_numpy() & es_b
            if i.sum() >= 30:
                row[apt] = rmse(p[i], y[i])
        rows[f"k={k}"] = row
    print(f"B satırı {len(b):,} (izli {es_b.sum():,})")
    print(pd.DataFrame(rows).T.round(1).to_string())
    # havalimanı × ay: SSE farkı (k=1), tutarlılık için
    p1 = np.maximum(60.0, base + rhat)
    g = pd.DataFrame({"apt": b["ADEP_mvt"], "ay": ay_b, "d": (p1 - y) ** 2 - (base - y) ** 2})[es_b]
    print("SSE farkı (M, k=1; negatif = iyi):")
    print((g.groupby(["apt", "ay"])["d"].sum() / 1e6).unstack("ay").round(2).to_string())
    # 10.10: cikti=<ek> → ADSB_B5_cv<ek>.parquet (aile karşılaştırması için ayrı dosyalar)
    pd.DataFrame({"MVT_ID_mvt": b["MVT_ID_mvt"], "rhat": rhat}).to_parquet(PRED / f"ADSB_B5_cv{opt.get('cikti', '')}.parquet",
                                                                          index=False)


def uretim(opt):
    ayarla(opt)
    k = float(opt.get("k", 1.0))
    apt = opt["apt"].split(",")
    a = a_veri()
    ref = damga_ref(a)
    # test Model A satırları (yerel/merkez komşuları için) — adsb_duzeltici.uretim ile aynı çerçeve
    bil = A.bilesenler()
    t = pd.read_parquet(A.PRED / f"TEST_{A.TEST_AD}.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "taksi"] + bil)
    t["taban_std"] = t[bil].std(axis=1, ddof=0)
    t = A.ikinci_test(t)
    t = t.merge(A.ozellikler(), on="MVT_ID_mvt", how="left", validate="one_to_one")
    t = t.merge(A.mvt_epoch(t["MVT_ID_mvt"]), on="MVT_ID_mvt", how="left", validate="one_to_one")
    t["taban"] = t["taksi"]
    t = t.merge(pd.read_parquet(config.DEP_RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"]), on="MVT_ID_mvt",
                how="left", validate="one_to_one")
    t = D.damga_hazirla(t)
    t = pd.concat([t, D.damga_te(t, ref, oof=False)], axis=1)
    ref5 = damga_ref5(a) if opt.get("damga5") == "1" else None
    if ref5 is not None:  # ortak_matris sütun kesişimi alıyor: test A çerçevesinde de olmalı
        t = pd.concat([t, D.damga_te(t, ref5, oof=False, onek="DAMGA5")], axis=1)
    if D.OPDI_OZ:  # aynı nedenle test A çerçevesine de
        t = D.opdi_ekle(t)
    if D.PARKB_OZ:
        t = D.parkb_ekle(t)
    if D.PLANLI_OZ:
        t = D.planli_ekle(t)
    # 05.10 akşam: kaynak=/guncel= ile taban ve güncel B dosyaları değiştirilebilir (Model B durum özellikleri sınaması)
    kaynak = pd.read_parquet(PRED / f"TEST_B_{opt.get('kaynak', 'tum6M_adsb')}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "taban"])
    rk = pd.read_parquet(config.DEP_RANK, columns=["MVT_ID_mvt", "MVT_MONTH"])
    bt = b_hazirla(kaynak.merge(rk, on="MVT_ID_mvt", how="left", validate="one_to_one"), ref, oof=False, ref5=ref5)
    D.TIP_KAT = None
    Xa, _ = ortak_matris(a, bt)
    if "tip" in Xa:
        D.TIP_KAT = Xa["tip"].cat.categories
    Xa, _ = ortak_matris(a, bt)
    _, Xb = ortak_matris(t, bt)
    Xb = Xb[list(Xa.columns)]
    es_a = D.satirlar(a)
    ra = (a["y_taksi"] - a["taban"]).to_numpy(float)
    modeller = [m for s in (42, 2026, 7) for m in D.egit(Xa[es_a], ra[es_a], PARAM, MODEL, seed=s)]
    et = D.satirlar(bt) & bt["ADEP_mvt"].isin(apt).to_numpy()
    r = np.zeros(len(bt))
    r[et] = D.tahmin(modeller, Xb[et])
    bt["taksi5"] = np.maximum(60.0, bt["taban"].to_numpy(float) + k * r)
    bt["rhat"] = r
    guncel = pd.read_parquet(PRED / f"TEST_B_{opt.get('guncel', 'tum6M_adsb_lirfte')}.parquet", columns=["MVT_ID_mvt", "taksi"])
    out = guncel.merge(bt[["MVT_ID_mvt", "ADEP_mvt", "taban", "taksi5", "rhat"]], on="MVT_ID_mvt", how="left",
                       validate="one_to_one")
    m = out["ADEP_mvt"].isin(apt).to_numpy()
    out["eski"] = out["taksi"]
    out["taksi"] = np.where(m, out["taksi5"], out["taksi"])
    ad = opt.get("ad", "tum6M_adsb5_lirfte")
    out[["MVT_ID_mvt", "ADEP_mvt", "taban", "taksi", "eski", "rhat"]].to_parquet(PRED / f"TEST_B_{ad}.parquet", index=False)
    f = (out["taksi"] - out["eski"]).to_numpy()
    print(f"test B satırı {len(out):,} | düzeltilen {int(et.sum()):,} | değişen {int((np.abs(f) > 1e-6).sum()):,} | "
          f"RMS fark {np.sqrt(np.mean(f[m] ** 2)):.0f} | ort fark {f[m].mean():.1f}")
    print(out.assign(f=f).groupby("ADEP_mvt")["f"].agg(["size", lambda s: int((s.abs() > 1e-6).sum()), "mean"]).round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    opt = dict(x.split("=", 1) for x in sys.argv[2:] if "=" in x)
    if "delta" in opt:  # 02.10: Huber δ (A düzelticisinde 400, 3 tohumda 600'den iyi)
        PARAM["alpha"] = float(opt["delta"])
    MODEL = opt.get("model", "lgb")
    D.NN_EPOK = int(opt.get("nn_epok", D.NN_EPOK))  # 10.10: model=…nn (TabM) için; A düzelticisiyle aynı anlam
    D.NN_W = float(opt["nn_w"]) if "nn_w" in opt else None
    {"cv": cv, "uretim": uretim}[sys.argv[1]](opt)

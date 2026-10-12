"""ADS-B düzeltici: Model A'nın (dondurulan v149, D_dw) taksi tahminini ADS-B özellikleriyle düzelten ikinci katman.

Neden: ADS-B'yi doğrudan tahmin olarak kullanmak 15.07.2025'te gün RMSE'sini 229,5 → 276,1 bozdu (büyük aykırılar,
havalimanına özgü BLOCK kayıt alışkanlıkları). Düzeltici, taban tahmin ile ADS-B arasındaki farkı ve iz kalitesini
görerek ne zaman ADS-B'ye güvenileceğini öğrenir.
Hedef: kalıntı r = gerçek taksi − taban taksi (yalnızca ADS-B eşleşmesi olan satırlar); tahmin = max(60, taban + k·r̂).
Doğrulama: Ocak 2025 ile eğit → Temmuz 2025'te ölç, ve tersi.
Taban: split_1 (Ocak+Temmuz dışarıda) harman tahmini, src/adsb.py d_dw_split.
Kullanım: python scripts/adsb_duzeltici.py cv [havalimani=0|1] [kayip=l2|huber]
"""
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config, validation as V  # noqa: E402

OZ = ["ADSB_ILK", "ADSB_HAREKET", "ADSB_TO_FARK", "ADSB_N", "ADSB_BOSLUK", "ADSB_BOSLUK5", "ADSB_ILK_GS", "ADSB_PARK",
      "ADSB_ILK_DURMA", "ADSB_ILK_UZAK", "ADSB_YOL", "ADSB_DURMA", "ADSB_KOSU", "ADSB_ESLESME"]
PARAM = dict(objective="l2", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=8)
TUR = 600


def veri(ek_aylar=False):
    """split_1 (Ocak+Temmuz 2025) satırları; ek_aylar=True ise split_2 (Şubat+Ağustos, kendi 10 aylık tabanıyla) da eklenir."""
    d = A.d_dw_split("split_1")
    if ek_aylar:
        parca = [d, A.d_dw_split("split_2")]
        if (A.PRED / "GCB_dw_lr03_ek5_l2_10_split_3.parquet").exists():  # kuyruk_22: Aralık+Haziran (30.09)
            parca.append(A.d_dw_split("split_3"))
        # 02.10: ek bölmeler. PRC_DUZ_S4=1 → split_4 (Kasım+Mart); PRC_DUZ_EK=split_4,split_5,...
        for s in os.environ.get("PRC_DUZ_EK", "split_4" if os.environ.get("PRC_DUZ_S4") == "1" else "").split(","):
            if s:
                parca.append(A.d_dw_split(s))
        d = pd.concat(parca, ignore_index=True)
    gunler = {(int(f.stem[8:10]), int(f.stem[11:13])) for f in A.ADSB_DIR.glob("oz_2025-*.parquet")}
    d = d[[(m, g) in gunler for m, g in zip(d["MVT_MONTH"], d["MVT_DAY"])]].reset_index(drop=True)
    d = d.merge(A.ozellikler(), on="MVT_ID_mvt", how="left", validate="one_to_one")  # yalnızca işlenmiş günler
    if YENI_OZ:
        d = d.merge(A.mvt_epoch(d["MVT_ID_mvt"]), on="MVT_ID_mvt", how="left", validate="one_to_one")
    if KUYRUK_OZ:
        d = kuyruk_ekle(d)
    if OPDI_OZ:
        d = opdi_ekle(d)
    if PLANLI_OZ:
        d = planli_ekle(d)
    if PARKB_OZ:
        d = parkb_ekle(d)
    d["gun"] = d["MVT_MONTH"].astype(str) + "-" + d["MVT_DAY"].astype(str)
    d["taban"] = A.taksi(d["NM_TAXI_DIFF_SEC"], d["ofset_taban"])
    d["y_taksi"] = d["NM_TAXI_DIFF_SEC"] - d["y"]
    return d


def kopyalar(opt, havalimani, X0):
    """seyrek=s30,s10 (01.10 akşam): ADS-B özellikleri seyreltilmiş izlerden yeniden hesaplanmış eğitim kopyaları
    (data/<ADSB_DIR>_<ek>, adsb_ozellik.py PRC_ADSB_SEYRELT). Her kopya kendi içinde hazırlanır (merkez, yerel, kayıt
    alışkanlığı); yalnız eğitime eklenir. Dönüş: [(X, r, ay)] (yalnız ADS-B eşleşmeli satırlar)."""
    out = []
    if not opt.get("seyrek"):
        return out
    asil = A.ADSB_DIR
    # opdi_satir=1: kopyada izi olmayan satır, orijinalinde de izsizse kopyayla birebir aynıdır (yalnız ağırlığı katlanır)
    # → kopyada yalnız orijinali ADS-B eşleşmeli satırlar tutulur (seyreltmeyle izini kaybedenler OPDI örneği olarak kalır)
    asil_es = None
    if OPDI_SATIR:
        o = A.ozellikler()
        asil_es = set(o.loc[o["ADSB_ESLESME"].fillna(0) > 0, "MVT_ID_mvt"])
    for ek in opt["seyrek"].split(","):
        A.ADSB_DIR = asil.parent / f"{asil.name}_{ek}"
        assert A.ADSB_DIR.exists(), A.ADSB_DIR
        try:
            dc = veri(opt.get("ek_aylar") == "1")
            if opt.get("oz") == "4":
                dc, _ = damga_ekle(dc)
            esc = satirlar(dc)
            if asil_es is not None:
                esc &= dc["MVT_ID_mvt"].isin(asil_es).to_numpy()
            Xc = matris(dc, havalimani)
        finally:
            A.ADSB_DIR = asil
        for c in X0.columns:
            if isinstance(X0[c].dtype, pd.CategoricalDtype):
                Xc[c] = pd.Categorical(Xc[c].astype(str), categories=X0[c].cat.categories)
        Xc = Xc[list(X0.columns)]
        rc = (dc["y_taksi"] - dc["taban"]).to_numpy(float)
        out.append((Xc[esc].reset_index(drop=True), rc[esc], dc["MVT_MONTH"].to_numpy()[esc]))
        print(f"kopya {ek}: {esc.sum():,} eşleşmeli satır (özgün tanımla)", flush=True)
    return out


def ekle_kopya(X, r, kop, aylar):
    """Eğitim matrisine seçilen aylardaki kopya satırlarını ekler."""
    if not kop:
        return X, r
    Xs, rs = [X], [r]
    for Xc, rc, ayc in kop:
        m = np.isin(ayc, list(aylar))
        Xs.append(Xc[m])
        rs.append(rc[m])
    return pd.concat(Xs, ignore_index=True), np.concatenate(rs)


def opdi_ekle(d):
    """opdi=1 (07.10): OPDI yer olayı özellikleri (data/opdi_oz.parquet)."""
    x = pd.read_parquet(config.DATA_DIR / "opdi_oz.parquet")
    x = x[["MVT_ID_mvt"] + [c for c in OPDI_KOL if c in x]]
    if OPDI_GUN:
        x = opdi_gun_pay().merge(x, on="MVT_ID_mvt", how="left")
    return d.merge(x, on="MVT_ID_mvt", how="left", validate="one_to_one")


def opdi_gun_pay():
    """opdi_gun=1 (07.10 gece, scripts/doluluk_denetim.py): OPDI kaynağında gün boşlukları var (2–10 ve 23–24 Oca 2026;
    2025'te ~30 gün, ör. 2–11 Eki). O gün o havalimanında OPDI ile eşleşen kalkış payı (etiketsiz, testte de hesaplanır):
    modelin "OPDI uçağı görmedi" ile "o gün OPDI verisi yok" durumlarını ayırması için."""
    k = pd.concat([pd.read_parquet(config.DATA_DIR / f, columns=["MVT_ID_mvt", "MVT_MONTH", "MVT_DAY", "ADEP_mvt"]).assign(yil=y)
                   for f, y in (("base_v126_train_v1.parquet", 2025), ("base_v126_rank_v1.parquet", 2026))], ignore_index=True)
    o = pd.read_parquet(config.DATA_DIR / "opdi_oz.parquet", columns=["MVT_ID_mvt", "OP_ESLESME"])
    k = k.merge(o, on="MVT_ID_mvt", how="left")
    k["OP_GUN_PAY"] = k["OP_ESLESME"].notna().astype(float).groupby(
        [k["yil"], k["MVT_MONTH"], k["MVT_DAY"], k["ADEP_mvt"]]).transform("mean")
    return k[["MVT_ID_mvt", "OP_GUN_PAY"]].drop_duplicates("MVT_ID_mvt")


def opdi_yerde(d):
    """OPDI uçağı kalkıştan en az 150 sn önce yerde (park / apron / taxiway) görüyor mu."""
    return (d["OP_YER"].to_numpy(float) >= 150) if "OP_YER" in d else np.zeros(len(d), bool)


def satirlar(d):
    """Düzelticinin eğitildiği / uygulandığı satırlar: ADS-B eşleşmeli; opdi_satir=1 ise OPDI'nin yerde gördükleri de."""
    es = d["ADSB_ESLESME"].fillna(0).to_numpy() > 0
    return es | opdi_yerde(d) if OPDI_SATIR else es


def opdi_merkez(fark, d):
    """Havalimanı × ay medyanından sapma (etiketsiz; testte de hesaplanır). En az 30 satır yoksa 0 kabul edilir."""
    grup = d["ADEP_mvt"].astype(str) + "_" + d["MVT_MONTH"].astype(str)
    med = fark.groupby(grup).agg(["median", "count"])
    return fark - grup.map(med["median"].where(med["count"] >= 30)).fillna(0.0).to_numpy(float)


def parkb_ekle(d):
    """parkb=1 (07.10 gece; scripts/park_b_ozellik.py): park yeri dizisi kalkış anına göre (kendi gelişi, sonraki varış,
    aynı park yerinden önceki / sonraki kalkış) ve ±15 dk NM'li kalkışların tahmini taksisi. Model B'de CV 844,7 → 775,2."""
    x = pd.read_parquet(config.DATA_DIR / "park_b_oz.parquet", columns=["MVT_ID_mvt"] + PARKB_KOL)
    return d.drop(columns=[c for c in PARKB_KOL if c in d]).merge(x, on="MVT_ID_mvt", how="left", validate="one_to_one")


def planli_ekle(d):
    """planli=1 (07.10, hata anatomisi): planlı saatler (X_DIFF = X − MVT, PRC_2/src/features.py:127-131). Gecikmeli
    uçuşlarda kayıt çoğu kez planlı saati (tam ya da ±15 dk) kopyalıyor; düzeltici bunu ADS-B hareketiyle kıyaslayabilsin."""
    k = ["MVT_ID_mvt", "SCHED_DIFF_SEC", "LOBT_DIFF_SEC", "EOBT_DIFF_SEC", "IOBT_DIFF_SEC"]
    x = pd.concat([pd.read_parquet(f, columns=k) for f in (config.DEP_TRAIN, config.DEP_RANK)]).drop_duplicates("MVT_ID_mvt")
    return d.drop(columns=[c for c in k[1:] if c in d]).merge(x, on="MVT_ID_mvt", how="left", validate="one_to_one")


def kuyruk_ekle(d):
    """kuyruk=1: NM off-block ↔ kalkış penceresi özellikleri (scripts/kuyruk_ozellik.py)."""
    kq = pd.read_parquet(config.DATA_DIR / "kuyruk_oz.parquet")
    d = d.merge(kq, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if PENCERE_OZ:  # pencere=1 (01.10): gecikme / pist / varış / METAR / ileri pencere özellikleri (Model A _kqgp ile aynı)
        for f in PENCERE_DOSYA:
            d = d.merge(pd.read_parquet(config.DATA_DIR / f), on="MVT_ID_mvt", how="left", validate="one_to_one")
    if DURUM_OZ:  # durum=1 (05.10 gece, denetim P1): Model A'ya LB'de kazandıran durum özellikleri düzelticiye de
        for f, cols in DURUM_DOSYA.items():
            x = pd.read_parquet(config.DATA_DIR / f)
            x = x[["MVT_ID_mvt"] + [c for c in x.columns if c.startswith(cols)]]
            d = d.merge(x, on="MVT_ID_mvt", how="left", validate="one_to_one")
    return d


def adsb_pencere(d, sure, anahtar):
    """(T − sure, T) aralığında aynı gruptan (havalimanı|pist) kalkan diğer uçak sayısı; sure NaN ise NaN."""
    t, s = d["mvt"].to_numpy(float), sure.to_numpy(float)
    out = np.full(len(d), np.nan)
    for _, idx in d.groupby(anahtar.to_numpy()).indices.items():
        ts = np.sort(t[idx])
        n = np.searchsorted(ts, t[idx], "left") - np.searchsorted(ts, t[idx] - s[idx], "right")
        out[idx] = np.where(np.isfinite(s[idx]), np.maximum(n, 0), np.nan)
    return out


def merkez(fark, d):
    """Havalimanı × ay medyanından sapma. Medyan, park yerinde görülen (ilk nokta < 3 kt) satırlardan; etiket kullanmaz,
    bu yüzden 2026'da test verisinin kendisinden hesaplanabilir (BLOCK kayıt alışkanlığı havalimanına göre değişiyor:
    15.07.2025 / 14.07.2026 denemesinde ADS-B − taban medyanı EHAM +30, EGLL ve LEMD ≈ −200 sn)."""
    iyi = (d["ADSB_ILK_GS"].fillna(99) < 3) & fark.notna()
    grup = d["ADEP_mvt"].astype(str) + "_" + d["MVT_MONTH"].astype(str)
    med = fark[iyi].groupby(grup[iyi]).agg(["median", "size"])
    genel = float(fark[iyi].median()) if iyi.any() else 0.0
    m = grup.map(med["median"].where(med["size"] >= 30)).fillna(genel)
    return fark - m


def yerel(d, dHm, pencere=1800.0, anahtar=None):
    """Aynı havalimanında ±pencere sn içindeki DİĞER uçuşların (park yerinde görülen) merkezlenmiş ADS-B − taban farkı
    ortalaması ve sayısı. O anki koşulları (kar, kuyruk) taşır; uçuşun kendi izi yoksa da tanımlıdır. Önek toplamlarıyla.
    anahtar: gruplama (varsayılan havalimanı; pist=1 için havalimanı|pist)."""
    out, say = np.full(len(d), np.nan), np.zeros(len(d))
    kal = ((d["ADSB_ILK_GS"].fillna(99) < 3) & dHm.notna()).to_numpy()
    v = np.where(kal, dHm.clip(-600, 600).fillna(0).to_numpy(), 0.0)
    c = kal.astype(float)
    t = d["mvt"].to_numpy(float)
    for _, idx in (d.groupby("ADEP_mvt") if anahtar is None else d.groupby(anahtar.to_numpy())).indices.items():
        o = idx[np.argsort(t[idx])]
        ts, vv, cc = t[o], v[o], c[o]
        sv, sc = np.r_[0.0, np.cumsum(vv)], np.r_[0.0, np.cumsum(cc)]
        lo, hi = np.searchsorted(ts, ts - pencere, "left"), np.searchsorted(ts, ts + pencere, "right")
        top, n = sv[hi] - sv[lo] - vv, sc[hi] - sc[lo] - cc
        out[o] = np.where(n > 0, top / np.maximum(n, 1), np.nan)
        say[o] = n
    return out, say


def matris(d, havalimani):
    X = pd.DataFrame({"taban": d["taban"], "nm": d["NM_TAXI_DIFF_SEC"]})
    for c in OZ:
        X[c] = d[c].astype(float)
    X["dH"] = d["ADSB_HAREKET"] - d["taban"]
    X["dI"] = d["ADSB_ILK"] - d["taban"]
    X["dH_m"] = merkez(X["dH"], d)
    X["dI_m"] = merkez(X["dI"], d)
    if YENI_OZ:
        X["yerel"], X["yerel_n"] = yerel(d, X["dH_m"])
        X["taban_std"] = d["taban_std"].astype(float)
        if PIST_OZ and "RUNWAY_mvt" in d:  # kuyruk piste özgü: aynı pistteki komşuların sapması
            X["yerel_pist"], X["yerel_pist_n"] = yerel(
                d, X["dH_m"], anahtar=d["ADEP_mvt"].astype(str) + "|" + d["RUNWAY_mvt"].astype(str))
        if YEREL15:  # daha dar pencere (±15 dk): ani yoğunluk değişimleri
            X["yerel15"], _ = yerel(d, X["dH_m"], pencere=900.0)
    if PARK_OZ:  # park yerinden uzaklıklar (oz=3): yolda ilk görülen uçuşlarda görülmeyen taksi kısmı
        global _PARK
        if _PARK is None:
            _PARK = A.park_tablosu()
        P = A.park_ozellik(d, _PARK)
        for c in P.columns:
            X[c] = P[c].to_numpy(float)
        X["PARK_ILK_S"] = X["PARK_ILK_M"] / 7.0  # ~14 kt taksi hızıyla görülmeyen kısmın kaba süresi
    if IZ5:  # iz5=1 (01.10): taksi başı ve park ayracı (adsb_ozellik.py PRC_ADSB_IZ5=1; veri klasörü adsb5)
        for c in OZ5:
            X[c] = d[c].astype(float) if c in d else np.nan
        X["dT4"] = X["ADSB_TAKSI_BAS"] - d["taban"]
        X["dPU"] = X["ADSB_PARK_U"] - d["taban"]
        X["dT4_m"] = merkez(X["dT4"], d)
        for c in ("DAMGA5_HY", "DAMGA5_PARK"):
            X[c] = d[c].to_numpy(float) if c in d else np.nan
    for c in ("DAMGA_HY", "DAMGA_PARK"):  # oz=4: kayıt alışkanlığı hedef kodlaması (cv/uretim'de eklenir)
        if c in d:
            X[c] = d[c].to_numpy(float)
    if TIP_OZ and "AIRCRAFT_TYPE_mvt" in d:  # itme süresi uçak tipine bağlı (NASA, pushback/ramp taxi)
        X["tip"] = pd.Categorical(d["AIRCRAFT_TYPE_mvt"].astype(str), categories=TIP_KAT) if TIP_KAT is not None             else d["AIRCRAFT_TYPE_mvt"].astype(str).astype("category")
    X["dNM"] = d["ADSB_HAREKET"] - d["NM_TAXI_DIFF_SEC"]
    if KUYRUK_OZ:
        for c in [c for c in d.columns if c.startswith("KQ_")]:
            X[c] = d[c].to_numpy(float)
        if PENCERE_OZ:
            for c in [c for c in d.columns if c.startswith(PENCERE_ONEK) or c.endswith("_SONRA_10")]:
                X[c] = d[c].to_numpy(float)
        if DURUM_OZ:
            for c in [c for c in d.columns if c.startswith(DURUM_ONEK)]:
                X[c] = d[c].to_numpy(float)
        if KUYRUK_ADSB and "RUNWAY_mvt" in d:  # ADS-B hareket başlangıcı ↔ kalkış arasında aynı pistten kalkan
            X["KQ_ADSB_PIST"] = adsb_pencere(d, d["ADSB_HAREKET"], d["ADEP_mvt"].astype(str) + "|" + d["RUNWAY_mvt"].astype(str))
            X["KQ_ILK_PIST"] = adsb_pencere(d, d["ADSB_ILK"], d["ADEP_mvt"].astype(str) + "|" + d["RUNWAY_mvt"].astype(str))
    if OPDI_OZ:  # opdi=1 (07.10): OPDI yer olayları (scripts/opdi_ozellik.py); olmayan satırda NaN
        for c in OPDI_KOL:
            X[c] = d[c].to_numpy(float) if c in d else np.nan
        for c in ("OP_PARK", "OP_TW", "OP_YER"):
            X["d" + c] = X[c] - d["taban"].to_numpy(float)
            X["d" + c + "_m"] = opdi_merkez(X["d" + c], d)
        if OPDI_GUN:
            X["OP_GUN_PAY"] = d["OP_GUN_PAY"].to_numpy(float)
    if PARKB_OZ:  # park yeri sınırları tabana göre: kendi gelişi (üst sınır), başka uçağın varışı (alt sınır); komşu taksi
        tb = d["taban"].to_numpy(float)
        for c in PARKB_KOL:
            X[c] = d[c].to_numpy(float) if c in d else np.nan
        for c in ("PB_AYNI", "PB_ON1", "PB_SON1", "PB_KT15", "PB_KT15_P"):
            X["d" + c] = X[c] - tb
    if PLANLI_OZ:  # MVT − planlı saat; ADS-B hareketine ve tabana göre farkı (kayıt planlı saati kopyalıyor mu)
        for c, ad in (("SCHED_DIFF_SEC", "S"), ("EOBT_DIFF_SEC", "E"), ("LOBT_DIFF_SEC", "L"), ("IOBT_DIFF_SEC", "I")):
            v = -d[c].to_numpy(float) if c in d else np.full(len(d), np.nan)
            X["PL_" + ad] = v
            X["PL_" + ad + "_H"] = v - d["ADSB_HAREKET"].to_numpy(float)
            X["PL_" + ad + "_T"] = v - d["taban"].to_numpy(float)
    if havalimani:
        X["apt"] = d["ADEP_mvt"].astype("category")
    if KAPSAMA_AT:  # kapsama yoğunluğu ("nerede/ne sıklıkla duyuldu") 2025→2026 kayıyor
        X = X.drop(columns=[c for c in ("ADSB_N", "yerel_n", "yerel_pist_n", "ADSB_BOSLUK", "ADSB_BOSLUK5") if c in X])
    if SADECE_MERKEZ:  # havalimanına özgü kayıt farkını taşıyan ham farklar çıkarılır (aktarım için)
        X = X.drop(columns=["dH", "dI", "dNM", "ADSB_HAREKET", "ADSB_ILK"])
    if NMSIZ:  # nmsiz=1 (06.10): NM'siz satırlarda NaN olan (LightGBM'de 0'a düşen) NM'ye bağlı girdiler atılır
        X = X.drop(columns=[c for c in ("nm", "taban_std", "dNM") if c in X])
        # 10.10: planlı saatlerden NM'ye bağlı olanlar (EOBT / LOBT / IOBT; B satırlarında boş) da; tarife (PL_S*) kalır
        X = X.drop(columns=[c for c in X.columns if c.startswith(("PL_E", "PL_L", "PL_I"))])
    return X


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def egit(X, r, param, model="lgb", seed=42):
    """Düzeltici modeli: lgb (varsayılan), cb (CatBoost, aynı kayıp), lgb+cb (iki ailenin basit ortalaması)."""
    modeller = []
    if "lgb" in model:
        modeller.append(lgb.train(dict(param, seed=seed), lgb.Dataset(X, r), TUR))
    if "cb" in model:
        from catboost import CatBoostRegressor
        kayip = "Huber:delta=600" if param.get("objective") == "huber" else "RMSE"
        Xc, kat = _cb_X(X)
        cb = CatBoostRegressor(loss_function=kayip, iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=10,
                               random_seed=seed, verbose=0, thread_count=8, cat_features=kat)
        cb.fit(Xc, r)
        modeller.append(cb)
    if "nn" in model:  # 10.10: TabM (scripts/nn_bilesen.py TabMReg; Model A tabanında harmana −1,2 kazandıran aile)
        import nn_bilesen as NN
        delta = float(param.get("alpha", 600)) if param.get("objective") == "huber" else 1e9
        delta = NN_DELTA or delta  # 10.10 gece: nn_delta=… ile TabM'in Huber δ'sı ayrı (yoksa LightGBM'inki)
        modeller.append(NN.TabMReg(epok=NN_EPOK, tohumlar=(seed,), delta=delta, tabm_k=NN_K, tabm_d=NN_D).fit(X, r))
    return modeller


def _cb_X(X):
    """08.10: CatBoost girdisi — kategorik sütunlar (ör. tip) kodlarıyla yerel kategori, diğerleri float
    (eski X.astype(float) kategorik 'tip' sütununda hata veriyordu; cb yolu daha önce hiç çalıştırılmamıştı)."""
    X = X.copy()
    kat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    for c in X.columns:
        X[c] = X[c].cat.codes.astype(int) if c in kat else X[c].astype(float)
    return X, kat


def tahmin(modeller, X):
    p = [m.predict(X) if isinstance(m, lgb.Booster) or getattr(m, "tabm", False) else m.predict(_cb_X(X)[0]) for m in modeller]
    nn = [getattr(m, "tabm", False) for m in modeller]
    if NN_W is not None and any(nn) and not all(nn):  # 10.10: TabM ailesinin ağırlığı (nn_w=…), öbür aileler eşit
        p_nn = np.mean([q for q, t in zip(p, nn) if t], axis=0)
        p_ot = np.mean([q for q, t in zip(p, nn) if not t], axis=0)
        return (1 - NN_W) * p_ot + NN_W * p_nn
    return np.mean(p, axis=0)


def cv(opt):
    t0 = time.time()
    havalimani = opt.get("havalimani", "1") == "1"
    param = dict(PARAM)
    if opt.get("kayip") == "huber":
        param.update(objective="huber", alpha=float(opt.get("delta", 600)))
    ek_aylar = opt.get("ek_aylar") == "1"
    d = veri(ek_aylar)
    if opt.get("oz") == "4":
        d, _ = damga_ekle(d)
    es = satirlar(d)
    X = matris(d, havalimani)
    r = (d["y_taksi"] - d["taban"]).to_numpy(float)
    y, base = d["y_taksi"].to_numpy(float), d["taban"].to_numpy(float)
    ay = d["MVT_MONTH"].to_numpy()
    rhat = np.zeros(len(d))
    apt = d["ADEP_mvt"].to_numpy()
    RA = (1, 7)  # raporlanan sınama ayları
    if opt.get("cv_tur") == "havalimani":  # görülmemiş havalimanına aktarım (2026'da EGLL/EDDM kapsaması yeni)
        katlar = [(f"{a} dışarıda", apt != a, apt == a) for a in sorted(set(apt[es]))]
    elif ek_aylar:  # Şubat+Ağustos iki katta da eğitimde; sınama yine Ocak/Temmuz (öncekilerle karşılaştırılabilir)
        # sinama=hepsi: her ay sırayla dışarıda (ör. LIRF'in izli olduğu Haziran/Ağustos/Aralık'ta örneklem dışı sınama)
        sinanan = sorted(set(ay.tolist())) if opt.get("sinama") == "hepsi" else (7, 1)
        if opt.get("sinama", "hepsi") != "hepsi":  # 07.10: sinama=11,12 → yalnız bu aylar sınanır ve raporlanır
            sinanan = tuple(int(x) for x in opt["sinama"].split(","))
            RA = tuple(sorted(sinanan))
        katlar = [(f"ay≠{b} → {b}", ay != b, ay == b) for b in sinanan]  # sınanan ay dışındaki tüm aylar
    elif (ay == 1).any() and (ay == 7).any():
        katlar = [(f"ay {a} → {b}", ay == a, ay == b) for a, b in ((1, 7), (7, 1))]
    else:  # tek ay varken geçici: tek/çift günler (gün katları iyimser; yalnızca ön bakış)
        tek = d["MVT_DAY"].to_numpy() % 2 == 1
        katlar = [("tek → çift gün", tek, ~tek), ("çift → tek gün", ~tek, tek)]
    kop = kopyalar(opt, havalimani, X)
    rk = {}  # egri=1 (04.10 gece): tur eğrisi; her 100 turda ay dışı tahmin (yalnız lgb)
    for ad, trm, tem in katlar:
        tr, te = es & trm, es & tem
        if tr.sum() < 1000 or te.sum() == 0:
            print(f"{ad}: yeterli satır yok ({tr.sum()}, {te.sum()})")
            continue
        Xe, re_ = ekle_kopya(X[tr].reset_index(drop=True), r[tr], kop, set(ay[tr].tolist()))
        ms = egit(Xe, re_, param, opt.get("model", "lgb"), seed=int(opt.get("cv_tohum", 42)))  # 11.10: cv_tohum (varsayılan 42 = eski)
        rhat[te] = tahmin(ms, X[te])
        if opt.get("egri") == "1":
            for kk in range(100, TUR + 1, 100):
                rk.setdefault(kk, np.zeros(len(d)))[te] = ms[0].predict(X[te], num_iteration=kk)
        print(f"{ad}: eğitim {tr.sum():,}, sınama {te.sum():,}", flush=True)
    if rk:
        w0 = V.month_weights((1, 7))
        for kk, v in rk.items():
            p = np.maximum(60.0, base + v)
            e = {a: np.mean((p[ay == a] - y[ay == a]) ** 2) for a in (1, 7)}
            print(f"egri tur={kk}: Oca {np.sqrt(e[1]):.3f} Tem {np.sqrt(e[7]):.3f} ağırlıklı "
                  f"{np.sqrt(w0[1] * e[1] + w0[7] * e[7]):.3f}", flush=True)
    if opt.get("yerel_model") == "1" and YENI_OZ:
        # kendi izi olmayan satırlar: yalnızca taban + yakın uçuşların sapması (yerel) ile ikinci düzeltici
        Xy = X[["taban", "nm", "yerel", "yerel_n", "taban_std"]]
        for ad, trm, tem in katlar:
            tr, te = trm & np.isfinite(X["yerel"].to_numpy()), tem & ~es & np.isfinite(X["yerel"].to_numpy())
            if tr.sum() < 1000 or te.sum() == 0:
                continue
            my = lgb.train(dict(param, num_leaves=15, min_data_in_leaf=500), lgb.Dataset(Xy[tr], r[tr]), 300)
            rhat[te] = my.predict(Xy[te])
            print(f"{ad} (izsiz satırlar): eğitim {tr.sum():,}, sınama {te.sum():,}", flush=True)
    w = V.month_weights((1, 7)) if RA == (1, 7) else {a: 1.0 / len(RA) for a in RA}
    if "apt" in opt:  # düzeltme yalnızca seçilen havalimanlarında uygulanır
        rhat = np.where(np.isin(apt, opt["apt"].split(",")), rhat, 0.0)
    if "tavan" in opt:
        rhat = np.clip(rhat, -float(opt["tavan"]), float(opt["tavan"]))
    rows = {}
    for k in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
        p = np.maximum(60.0, base + k * rhat)
        row = {}
        for a in RA:
            mm = ay == a
            if mm.any():
                row[f"ay {a}"] = rmse(p[mm], y[mm])
                row[f"ay {a} ADS-B satırı"] = rmse(p[mm & es], y[mm & es])
        row["Oca/Tem ağırlıklı"] = float(np.sqrt(sum(w[a] * np.mean((p[ay == a] - y[ay == a]) ** 2)
                                                 for a in RA if (ay == a).any()) / sum(
            w[a] for a in RA if (ay == a).any())))
        rows[f"k={k}"] = row
    print(f"satır {len(d):,} (ADS-B eşleşmeli {es.sum():,}); günler: Ocak {d.loc[ay == 1, 'gun'].nunique()}, "
          f"Temmuz {d.loc[ay == 7, 'gun'].nunique()}")
    print(pd.DataFrame(rows).T.round(2).to_string())
    k1 = np.maximum(60.0, base + rhat)
    tab = {}
    for apt, i in d[np.isin(ay, RA)].groupby("ADEP_mvt").indices.items():
        i = np.flatnonzero(np.isin(ay, RA))[i]
        tab[apt] = {"satır": len(i), "ADS-B %": 100 * es[i].mean(), "taban": rmse(base[i], y[i]),
                    "düzeltilmiş": rmse(k1[i], y[i])}
    print(pd.DataFrame(tab).T.round(1).to_string())
    if OPDI_OZ:  # OPDI hedef satırları: ADS-B izi yok ya da yalnız kalkış koşusu (ilk nokta < 150 sn), OPDI yerde görüyor
        zayif = (~(d["ADSB_ESLESME"].fillna(0).to_numpy() > 0)) | (d["ADSB_ILK"].fillna(0).to_numpy(float) < 150)
        h = opdi_yerde(d) & zayif & np.isin(ay, RA)
        tab = {}
        for a_, i in d[h].groupby("ADEP_mvt").indices.items():
            i = np.flatnonzero(h)[i]
            tab[a_] = {"satır": len(i), "taban": rmse(base[i], y[i]), "düzeltilmiş": rmse(k1[i], y[i]),
                       "ΔSSE (M)": (np.sum((k1[i] - y[i]) ** 2) - np.sum((base[i] - y[i]) ** 2)) / 1e6}
        print(f"OPDI hedef satırları (Oca/Tem): {h.sum():,} | taban {rmse(base[h], y[h]):.1f} → {rmse(k1[h], y[h]):.1f}")
        print(pd.DataFrame(tab).T.round(1).to_string())
    print(f"süre {time.time() - t0:.0f} sn")
    pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "rhat": rhat}).to_parquet(
        config.DATA_DIR / "preds" / f"ADSB_duz_cv_h{int(havalimani)}_{opt.get('kayip', 'l2')}.parquet", index=False)


def uretim(opt):
    """2025 Ocak+Temmuz'un tüm ADS-B satırlarıyla eğit; 2026 test satırlarına uygula → TEST_D_dw_adsb<ek>.parquet."""
    havalimani = opt.get("havalimani", "1") == "1"
    k = float(opt.get("k", 1.0))
    param = dict(PARAM)
    if opt.get("kayip") == "huber":
        param.update(objective="huber", alpha=float(opt.get("delta", 600)))
    global TIP_KAT
    d = veri(opt.get("ek_aylar") == "1")
    bil = A.bilesenler()
    t = pd.read_parquet(A.PRED / f"TEST_{A.TEST_AD}.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "harman", "taksi"] + bil)
    t["taban_std"] = t[bil].std(axis=1, ddof=0)
    t = A.ikinci_test(t)
    t = t.merge(A.ozellikler(), on="MVT_ID_mvt", how="left", validate="one_to_one")
    if YENI_OZ:
        t = t.merge(A.mvt_epoch(t["MVT_ID_mvt"]), on="MVT_ID_mvt", how="left", validate="one_to_one")
    t["taban"] = t["taksi"]
    t = t.merge(pd.read_parquet(config.DEP_RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"]), on="MVT_ID_mvt",
                how="left", validate="one_to_one")
    if KUYRUK_OZ:
        t = kuyruk_ekle(t)
    if OPDI_OZ:
        t = opdi_ekle(t)
    if PLANLI_OZ:
        t = planli_ekle(t)
    if PARKB_OZ:
        t = parkb_ekle(t)
    if opt.get("oz") == "4":  # kayıt alışkanlığı: eğitimde ay dışı, testte tüm eğitimden
        d, t = damga_ekle(d, t)
    es = satirlar(d)
    X = matris(d, havalimani)
    if "tip" in X:
        TIP_KAT = X["tip"].cat.categories
    r = (d["y_taksi"] - d["taban"]).to_numpy(float)
    Xe, re_ = ekle_kopya(X[es].reset_index(drop=True), r[es], kopyalar(opt, havalimani, X), set(d["MVT_MONTH"].tolist()))
    tohumlar = [int(s) for s in opt.get("tohumlar", "42,2026,7").split(",")]  # 11.10: ek tohum üretimi (scripts/duz_tohum_birlestir.py)
    modeller = [m for s in tohumlar for m in egit(Xe, re_, param, opt.get("model", "lgb"), seed=s)]
    et = satirlar(t)
    Xt = matris(t, havalimani)
    assert list(Xt.columns) == list(X.columns), (list(X.columns), list(Xt.columns))
    if havalimani:
        Xt["apt"] = pd.Categorical(t["ADEP_mvt"], categories=X["apt"].cat.categories)
    rhat = np.zeros(len(t))
    rhat[et] = tahmin(modeller, Xt[et])
    bil = {}
    if opt.get("bilesen") == "1":  # 11.10: aile bileşenleri ayrı (LightGBM / TabM ortalaması) → v yeniden eğitimsiz değişebilir
        nn = [getattr(m, "tabm", False) for m in modeller]
        for ad, sec in (("rhat_lgb", [not x for x in nn]), ("rhat_nn", nn)):
            v = np.zeros(len(t))
            if any(sec):
                v[et] = tahmin([m for m, s_ in zip(modeller, sec) if s_], Xt[et])
            bil[ad] = v
    if "apt" in opt:  # düzeltme yalnızca seçilen havalimanlarında
        rhat = np.where(t["ADEP_mvt"].isin(opt["apt"].split(",")).to_numpy(), rhat, 0.0)
    if "tavan" in opt:
        rhat = np.clip(rhat, -float(opt["tavan"]), float(opt["tavan"]))
    t["rhat"] = rhat
    t["taksi_taban"] = t["taban"]
    t["taksi"] = np.maximum(60.0, t["taban"] + k * rhat)
    if "yanki_beta" in opt:  # tarife yankısı hedge'ini geri ekle (Huber düzeltici siliyor; 30.09 LIRF analizi)
        b = float(opt["yanki_beta"])
        e1 = pd.read_parquet(A.PRED / f"TEST_{A.E1_AD}.parquet", columns=["MVT_ID_mvt", "p_kilit"])
        pk = t[["MVT_ID_mvt"]].merge(e1, on="MVT_ID_mvt", how="left", validate="one_to_one")["p_kilit"].fillna(0).to_numpy()
        ham = pd.read_parquet(config.RANK_RAW,
                              columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
        ham["gec"] = (ham["MVT_TIME_UTC_mvt"] - ham["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
        g = np.maximum(60.0, t[["MVT_ID_mvt"]].merge(ham[["MVT_ID_mvt", "gec"]], on="MVT_ID_mvt", how="left",
                                                     validate="one_to_one")["gec"].fillna(0).to_numpy(float))
        m = et & t["ADEP_mvt"].isin(opt.get("yanki_apt", "LIRF").split(",")).to_numpy()
        p = t["taksi"].to_numpy(float)
        t["taksi"] = np.where(m, np.maximum(60.0, p + b * pk * (g - p)), p)
        print(f"yankı hedge'i: β={b}, {m.sum():,} satır | ort. değişim {(t['taksi'].to_numpy() - p)[m].mean():.1f} sn | "
              f"ort. p_kilit {pk[m].mean():.3f}")
    ek = opt.get("ek", "")
    out = t[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "harman", "taksi", "taksi_taban", "rhat", "ADSB_ESLESME", "ADEP_mvt"]]
    if bil:
        out = out.assign(**bil)
    out.to_parquet(A.PRED / f"TEST_{A.TEST_AD}_adsb{ek}.parquet", index=False)
    print(f"eğitim satırı {es.sum():,} | test satırı {len(t):,}, ADS-B eşleşmeli {et.sum():,} (%{100 * et.mean():.1f})")
    g = t.assign(fark=t["taksi"] - t["taksi_taban"]).groupby("ADEP_mvt").agg(
        satir=("fark", "size"), adsb=("ADSB_ESLESME", lambda s: 100 * (s.fillna(0) > 0).mean()),
        ort_fark=("fark", "mean"), rms_fark=("fark", lambda s: float(np.sqrt(np.mean(s ** 2)))))
    print(g.round(1).to_string())


def damga_te(d, ref, oof=True, m=30.0, onek="DAMGA"):
    """(havalimanı, havayolu) ve (havalimanı, park yeri) başına u = gerçek taksi − ADS-B hareket süresi ortalaması
    (Bayes, öncül havalimanı ortalaması). ref: etiketli ve park yerinde görülen satırlar. oof=True: kendi ayı dışlanır."""
    out = pd.DataFrame(index=d.index)
    for ad, anahtar in (("HY", "AIRLINE_LETTERS"), ("PARK", "STAND_mvt")):
        k_d = d["ADEP_mvt"].astype(str) + "|" + d[anahtar].astype(str)
        k_r = ref["ADEP_mvt"].astype(str) + "|" + ref[anahtar].astype(str)
        v = np.full(len(d), np.nan)
        aylar = np.unique(d["MVT_MONTH"]) if oof else [None]
        for ay in aylar:
            rr = ref if ay is None else ref[ref["MVT_MONTH"] != ay]
            kr = k_r.loc[rr.index]
            onc = rr.groupby("ADEP_mvt")["u"].mean()
            g = rr.groupby(kr)["u"].agg(["sum", "size"])
            i = np.ones(len(d), bool) if ay is None else (d["MVT_MONTH"] == ay).to_numpy()
            kd = k_d[i]
            o = d.loc[i, "ADEP_mvt"].map(onc).to_numpy(float)
            s_, n_ = kd.map(g["sum"]).fillna(0).to_numpy(float), kd.map(g["size"]).fillna(0).to_numpy(float)
            v[i] = (s_ + m * o) / (n_ + m)
        out[f"{onek}_{ad}"] = v
    return out


def damga_hazirla(d):
    """Eğitim (2025) satırları için u ve anahtarlar; ref = park yerinde görülen, hedef < 2 sa."""
    ek = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "AIRLINE_LETTERS", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
                                                   "RUNWAY_mvt"])
                    for f in (config.DEP_TRAIN, config.DEP_RANK)]).drop_duplicates("MVT_ID_mvt")
    d = d.merge(ek, on="MVT_ID_mvt", how="left")
    return d


def damga_ekle(d, t=None):
    """d (eğitim) satırlarına ay dışı, t (test) satırlarına tüm eğitimden DAMGA_* sütunları."""
    d = damga_hazirla(d)
    iyi = ((d["ADSB_ESLESME"].fillna(0) > 0) & (d["ADSB_ILK_GS"].fillna(99) < 3) & d["ADSB_HAREKET"].notna()
           & (d["y_taksi"] < 7200))
    ref = d[iyi].assign(u=d.loc[iyi, "y_taksi"] - d.loc[iyi, "ADSB_HAREKET"])
    d = pd.concat([d, damga_te(d, ref, oof=True)], axis=1)
    ref5 = None
    if IZ5 and "ADSB_PARK_U" in d:  # iz5=1: kayıt alışkanlığı park ayracıyla (dar ayraç ≤ 120 sn; 15_arastirma I-c)
        iyi5 = ((d["ADSB_ESLESME"].fillna(0) > 0) & d["ADSB_PARK_U"].notna() & (d["ADSB_PARK_ARALIK"] <= 120)
                & (d["y_taksi"] < 7200))
        ref5 = d[iyi5].assign(u=d.loc[iyi5, "y_taksi"] - d.loc[iyi5, "ADSB_PARK_U"])
        d = pd.concat([d, damga_te(d, ref5, oof=True, onek="DAMGA5")], axis=1)
    if t is not None:
        t = damga_hazirla(t)
        t = pd.concat([t, damga_te(t, ref, oof=False)], axis=1)
        if ref5 is not None:
            t = pd.concat([t, damga_te(t, ref5, oof=False, onek="DAMGA5")], axis=1)
    return d, t


KAPSAMA_AT = False  # kapsama=0: kapsama yoğunluğu özellikleri atılır
IZ5 = False  # iz5=1: taksi başı + park ayracı + ayraçlı kayıt alışkanlığı (01.10)
OZ5 = ["ADSB_TAKSI_BAS", "ADSB_PARK_L", "ADSB_PARK_U", "ADSB_PARK_ARALIK", "ADSB_PARK_ILK"]
KUYRUK_OZ = False  # kuyruk=1: kuyruk / pencere özellikleri (30.09)
PENCERE_OZ = False  # pencere=1 (kuyruk=1 ile): Model A _kqgp'nin ek özellikleri (01.10)
PENCERE_DOSYA = ("gecikme_oz.parquet", "gecikme_pist_oz.parquet", "varis_wx_oz.parquet", "pencere_ileri_oz.parquet",
                 "park_varis_oz.parquet")
PENCERE_ONEK = ("GEC_", "AR_", "WX_", "GECC_", "GECF_", "NMT_", "IN_", "NX_")
# durum=1 (kuyruk=1 ile): komşu aralıkları (KOM_T / KOM_PIST yardımcıları hariç), komşuların tahmini taksisi (kqgpxg OOF /
# TEST_D_kqgpxg; düzelticinin OOF tabanıyla aynı kurgu), günlük seriler, pist konfigürasyonu, yankı geçmişi
DURUM_OZ = False
DURUM_DOSYA = {"komsu_oz.parquet": ("KOM_PP_", "KOM_PS_", "KOM_AP_", "KOM_AS_"), "komsu_tah_oz.parquet": ("KTAH_",),
               "gunluk_oz.parquet": ("GS_",), "pist_oz.parquet": ("PC_",), "yanki_te_oz.parquet": ("YTE_",)}
DURUM_ONEK = ("KOM_PP_", "KOM_PS_", "KOM_AP_", "KOM_AS_", "KTAH_", "GS_", "PC_", "YTE_")
KUYRUK_ADSB = False  # kuyruk_adsb=1: ADS-B hareket / ilk görülme ↔ kalkış penceresinde aynı pistten kalkış sayısı
PIST_OZ = False  # pist=1: aynı pistteki komşuların yerel sapması (oz=4 ile; RUNWAY_mvt damga_hazirla'dan)
YEREL15 = False  # yerel15=1: ±15 dk pencereli yerel sapma
TIP_OZ = False  # tip=1: uçak tipi kategorik
TIP_KAT = None
SADECE_MERKEZ = False
NMSIZ = False  # nmsiz=1: nm / taban_std / dNM atılır (B5 için; adsb_b_duzeltici5.ayarla)
OPDI_OZ = False  # opdi=1 (07.10): OPDI yer olayı özellikleri
PARKB_OZ = False  # parkb=1 (07.10 gece): park yeri dizisi + komşu taksi (scripts/park_b_ozellik.py)
PARKB_KOL = ["PB_ON1", "PB_ON1_HY", "PB_ON1_TIP", "PB_AYNI", "PB_SON1", "PD_ON1", "PD_SON1", "PB_KT15", "PB_KT15_P"]
PLANLI_OZ = False  # planli=1 (07.10): planlı saatler (tarife / EOBT / LOBT / IOBT) ve ADS-B hareketine göre farkları
OPDI_GUN = False  # opdi_gun=1 (07.10 gece): o gün o havalimanında OPDI eşleşme payı (kaynak gün boşlukları)
NN_EPOK = 8  # 10.10: model=…nn için TabM epok sayısı (nn_epok=…)
NN_W = None  # 10.10: model=lgb+nn için TabM ağırlığı (nn_w=…; yoksa eşit ortalama)
NN_K, NN_D = 8, 512  # 10.10 akşam: düzelticideki TabM'in üye sayısı / blok genişliği (nn_k=…, nn_d=…)
NN_DELTA = None  # 10.10 gece: düzelticideki TabM'in Huber δ'sı (nn_delta=…; yoksa LightGBM'in alpha'sı)
OPDI_SATIR = False  # opdi_satir=1: ADS-B izi olmayan ama OPDI'nin yerde gördüğü satırlar da eğitilir / düzeltilir
OPDI_KOL = ["OP_ESLESME", "OP_ILK", "OP_SON", "OP_YER", "OP_PARK", "OP_PARK_SURE", "OP_PARK_SON", "OP_APRON", "OP_TW",
            "OP_PIST", "OP_PIST_SURE", "OP_TW_TOPLAM", "OP_TW_ENUZUN", "OP_TW_SAY", "OP_N", "OP_PARK_UYUM", "OP_PIST_UYUM"]
YENI_OZ = False  # yerel sapma (±30 dk diğer uçuşlar) ve taban bileşen std'si; oz=2 ile açılır
PARK_OZ = False  # park yeri uzaklıkları; oz=3 ile (oz=2'yi de içerir)
_PARK = None

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    SADECE_MERKEZ = opt.get("merkez") == "1"
    YENI_OZ = opt.get("oz") in ("2", "3", "4")
    PARK_OZ = opt.get("oz") in ("3", "4")
    TIP_OZ = opt.get("tip") == "1"
    KAPSAMA_AT = opt.get("kapsama") == "0"
    PIST_OZ = opt.get("pist") == "1"
    KUYRUK_OZ = opt.get("kuyruk") == "1"
    PENCERE_OZ = opt.get("pencere") == "1"
    IZ5 = opt.get("iz5") == "1"
    assert not IZ5 or "adsb5" in str(A.ADSB_DIR), "iz5=1, PRC_ADSB_OZ=adsb5 ile kullanılır"
    assert not PENCERE_OZ or KUYRUK_OZ, "pencere=1, kuyruk=1 ile kullanılır"
    DURUM_OZ = opt.get("durum") == "1"
    OPDI_OZ = opt.get("opdi") == "1"
    PLANLI_OZ = opt.get("planli") == "1"
    PARKB_OZ = opt.get("parkb") == "1"
    OPDI_SATIR = opt.get("opdi_satir") == "1"
    assert not OPDI_SATIR or OPDI_OZ, "opdi_satir=1, opdi=1 ile kullanılır"
    OPDI_GUN = opt.get("opdi_gun") == "1"
    assert not OPDI_GUN or OPDI_OZ, "opdi_gun=1, opdi=1 ile kullanılır"
    assert not DURUM_OZ or KUYRUK_OZ, "durum=1, kuyruk=1 ile kullanılır"
    KUYRUK_ADSB = opt.get("kuyruk_adsb") == "1"
    YEREL15 = opt.get("yerel15") == "1"
    for a, v in list(opt.items()):  # p:<lgb parametresi>=<değer>, tur=<n>
        if a.startswith("p:"):
            if v in ("True", "False"):  # ör. p:linear_tree=True (doğrusal yaprak uç değerleri dışa taşır)
                PARAM[a[2:]] = v == "True"
            else:
                PARAM[a[2:]] = (type(PARAM[a[2:]])(v) if a[2:] in PARAM
                                else int(v) if v.lstrip("-").isdigit() else float(v))  # ör. p:bagging_seed=7 tamsayı ister
    TUR = int(opt.get("tur", TUR))
    NN_EPOK = int(opt.get("nn_epok", NN_EPOK))
    NN_W = float(opt["nn_w"]) if "nn_w" in opt else None
    NN_K, NN_D = int(opt.get("nn_k", NN_K)), int(opt.get("nn_d", NN_D))
    NN_DELTA = float(opt["nn_delta"]) if "nn_delta" in opt else None
    {"cv": cv, "uretim": uretim}[sys.argv[1]](opt)

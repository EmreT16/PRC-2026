"""v126 YENİ_200 Model A özellik hattının bu depodaki sürümü.

Kaynak: PRC_4\\src\\train_and_build_v126_leaves200.py:47-217.

Bilinçli farklar:
- Hedef kodlamalar (TE v2'nin 9 sütunu + V3'ün STAND_RWY_TE_OFFSET'i) dosyadan okunmaz; her kat için
  validation.grouped_oof_te ile yeniden hesaplanır. TE v2 12 ay üzerinde rastgele katla üretildiği için
  doğrulama aylarına sızıyor.
- Kayan sayaçlar temizlik öncesi tablo üzerinde hesaplanır (v126 temizlik sonrası; fark 453 satır).
- pandas 3: metin anahtarlarda fillna("nan") (eski astype(str) davranışı).
v126'nın hataları matrix(..., v126_bugs=True) ile korunur: metin sütunları → 0 ve NaN → 0.
"""
import os
import warnings

import numpy as np
import pandas as pd

from . import config, validation

PRC4_FEAT = config.PRC4_DIR / "data" / "features"
V3_FILES = {"train": PRC4_FEAT / "advanced_features_train_v3.parquet",
            "rank": (config.DATA_DIR if config.FINAL else PRC4_FEAT) / "advanced_features_rank_v3.parquet"}  # final: final_taban.py
DIST_FILE = PRC4_FEAT / "stand_rwy_distances.csv"
V3_TE_COL = "STAND_RWY_TE_OFFSET"      # V3'teki hedef tabanlı sütun: dosyadan alınmaz
NOISY_STANDS = {"102", "105", "824", "826", "828", "829", "907", "HNG"}
# float32'ye indirilmeyecekler: kimlik (~1,8·10⁸; float32 ancak ~1,7·10⁷'ye kadar tam sayıyı kesin tutar) ve hedefler
KEEP64 = {"MVT_ID_mvt", "NM_OFFSET_SEC", "NM_TAXI_DIFF_SEC", "TAXITIME_SEC_mvt"}

CAT_V126 = ["AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt"]
EXCLUDE_V126 = {"MVT_ID_mvt", "TAXITIME_SEC_mvt", "NM_OFFSET_SEC", "NM_TAXI_DIFF_SEC",
                "ADEP_mvt", "VISIBILITY", "IS_LVP", "IS_LOW_VIS"}


# ----------------------------------------------------------------------------- temel tablo (hedefsiz)
def base_table(which, cache=True):
    """Tüm kalkış satırları için hedef kodlama dışındaki v126 özellikleri. which: 'train' | 'rank'."""
    path = config.DATA_DIR / f"base_v126_{which}_v1.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    src = config.DEP_TRAIN if which == "train" else config.DEP_RANK
    df = pd.read_parquet(src)
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in KEEP64:
            df[c] = df[c].astype(np.float32)
    v3 = pd.read_parquet(V3_FILES[which]).drop(columns=[V3_TE_COL])
    df = df.merge(v3, on="MVT_ID_mvt", how="left", validate="one_to_one")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # sütun sütun ekleme uyarısı
        add_distances(df)
        add_cdm(df)
        add_noisy(df)
        add_rolling(df, year=2025 if which == "train" else 2026)
    df = df.copy()
    if cache:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


def add_distances(df):
    """OSM stand–pist mesafeleri (v126:75-108); boşluklar tablonun kendi havalimanı medyanıyla (v126 gibi)."""
    dist = pd.read_csv(DIST_FILE)
    dist["STAND_str"] = dist["STAND_ID"].astype(str).str.strip().str.upper()
    dist["RWY_str"] = dist["RUNWAY"].astype(str).str.strip().str.upper().str.lstrip("0")
    min_d = dist.groupby(["ADEP", "STAND_str"])["DIST_M"].min().rename("STAND_MIN_DIST_M").reset_index()
    rwy_d = (dist[["ADEP", "STAND_str", "RWY_str", "DIST_M"]]
             .drop_duplicates(subset=["ADEP", "STAND_str", "RWY_str"])
             .rename(columns={"DIST_M": "STAND_TO_RWY_DIST_M"}))
    keys = pd.DataFrame({"ADEP": df["ADEP_mvt"].to_numpy(),
                         "STAND_str": df["STAND_mvt"].fillna("nan").astype(str).str.strip().str.upper().to_numpy(),
                         "RWY_str": df["RUNWAY_mvt"].fillna("nan").astype(str).str.strip().str.upper()
                         .str.lstrip("0").to_numpy()})
    df["STAND_TO_RWY_DIST_M"] = keys.merge(rwy_d, on=["ADEP", "STAND_str", "RWY_str"], how="left")[
        "STAND_TO_RWY_DIST_M"].to_numpy()
    df["STAND_MIN_DIST_M"] = keys.merge(min_d, on=["ADEP", "STAND_str"], how="left")["STAND_MIN_DIST_M"].to_numpy()
    for col, default in (("STAND_TO_RWY_DIST_M", 2200.0), ("STAND_MIN_DIST_M", 1000.0)):
        med = df.groupby("ADEP_mvt")[col].transform("median").fillna(default)
        df[col] = df[col].fillna(med).astype(np.float32)
    df["STAND_DIST_RATIO"] = (df["STAND_TO_RWY_DIST_M"] / (df["STAND_MIN_DIST_M"] + 50.0)).clip(1.0, 10.0).astype(
        np.float32)


def add_cdm(df):
    """Sızıntısız CDM etkileşimleri (v126:114-136)."""
    z = lambda c: df[c].fillna(0).astype(np.float32)  # noqa: E731
    ae, q30, q15, qd = z("AOBT_MINUS_EOBT"), z("Q_RWY_DEP_30"), z("Q_RWY_DEP_15"), z("QUEUE_DELAY_EST")
    ad, ed, ld, idf, pk = z("AOBT_DIFF_SEC"), z("EOBT_DIFF_SEC"), z("LOBT_DIFF_SEC"), z("IOBT_DIFF_SEC"), z("IS_PEAK_HOUR")
    df["AOBT_EOBT_QUEUE_PROD"] = ae * q30
    df["AOBT_EOBT_QUEUE15_PROD"] = ae * q15
    df["QUEUE_DELAY_AOBT_SUM"] = qd + ae.clip(0, None)
    df["CDM_REVISION_MAGNITUDE"] = (ld - ed).abs()
    df["LOBT_IOBT_DIFF"] = ld - idf
    df["EOBT_AOBT_DIFF"] = ed - ad
    df["PEAK_AOBT_EOBT"] = pk * ae
    df["AOBT_EOBT_ABS"] = ae.abs()
    df["QUEUE_CDM_RATIO"] = (q30 / (ae.abs() + 1.0)).clip(0, 100)


def add_noisy(df):
    """LIRF gürültülü stand bayrakları (v126:150-160)."""
    lirf = df["ADEP_mvt"].eq("LIRF")
    noisy = df["STAND_mvt"].isin(NOISY_STANDS)
    df["LIRF_NOISY_STAND"] = (lirf & noisy).astype(np.float32)
    df["LIRF_NOISY_16R"] = (lirf & noisy & df["RUNWAY_mvt"].eq("16R")).astype(np.float32)


def add_rolling(df, year):
    """Son 10/30/60 dk'da havalimanında ve son 10/30 dk'da aynı pistte kalkan uçak sayısı (v126:162-190)."""
    ts = (pd.to_datetime(pd.DataFrame({"year": year, "month": df["MVT_MONTH"], "day": df["MVT_DAY"]}))
          + pd.to_timedelta(df["MVT_MINOFDAY_UTC"], unit="m")).to_numpy("datetime64[ns]")
    for keys, wins, prefix in ((["ADEP_mvt"], (10, 30, 60), "DEP_COUNT_"),
                               (["ADEP_mvt", "RUNWAY_mvt"], (10, 30), "DEP_RWY_COUNT_")):
        groups = df.groupby(keys, sort=False).indices
        for w in wins:
            out = np.zeros(len(df), dtype=np.float32)
            win = np.timedelta64(w, "m")
            for idx in groups.values():
                t = ts[idx]
                s = np.sort(t)
                out[idx] = np.searchsorted(s, t, side="left") - np.searchsorted(s, t - win, side="left")
            df[f"{prefix}{w}"] = out


# ----------------------------------------------------------------------------- hedef kodlama
def _up(s):
    return s.fillna("nan").astype(str).str.strip().str.upper()


def te_keys(df):
    """TE v2 (build_advanced_target_encodings_v2.py:36-51) ve V3 (feature_engineering_v3.py:115-116) anahtarları."""
    apt = df["ADEP_mvt"].astype(str)
    stand, rwy = _up(df["STAND_mvt"]), _up(df["RUNWAY_mvt"])
    return {
        "STAND_RWY": apt + "_" + stand + "_" + rwy,
        "OP_APT": apt + "_" + _up(df["AIRCRAFT_OPERATOR_flt"]),
        "ROUTE": apt + "_" + _up(df["ADES_mvt"]),
        "STAND_APT": apt + "_" + stand,
        "RWY_APT": apt + "_" + rwy,
        "ACTYPE_APT": apt + "_" + _up(df["AIRCRAFT_TYPE_mvt"]),
        "PIER_RWY": apt + "_" + stand.str.slice(0, 2) + "_" + rwy,
        "HOUR_APT": apt + "_" + df["LOCAL_HOUR"].astype(str),
        "RWY_STAND_RAW": (apt + "_" + df["RUNWAY_mvt"].fillna("nan").astype(str) + "_"
                          + df["STAND_mvt"].fillna("nan").astype(str)),
    }


# (çıktı sütunu, anahtar, yumuşatma m, istatistik)
TE_SPECS = [
    ("STAND_RWY_TE", "STAND_RWY", 20.0, "mean"),
    ("OP_APT_TE", "OP_APT", 20.0, "mean"),
    ("ROUTE_TE", "ROUTE", 20.0, "mean"),
    ("STAND_APT_TE", "STAND_APT", 25.0, "mean"),
    ("RWY_APT_TE", "RWY_APT", 30.0, "mean"),
    ("ACTYPE_APT_TE", "ACTYPE_APT", 20.0, "mean"),
    ("PIER_RWY_TE", "PIER_RWY", 30.0, "mean"),
    ("HOUR_APT_TE", "HOUR_APT", 50.0, "mean"),
    ("STAND_RWY_STD_TE", "STAND_RWY", 20.0, "std"),
    (V3_TE_COL, "RWY_STAND_RAW", 15.0, "mean"),
]
TE_COLS = [s[0] for s in TE_SPECS]


def add_te(train, apply, groups_train, insample=False):
    """train satırlarına gruplu OOF, apply satırlarına train'in tamamından TE sütunlarını yazar (yerinde).

    insample=True: train satırlarının TE'si de kendi hedefleri dahil tüm train'den (A2 kalibrasyonu:
    v26'daki eğitim içi TE hatasının benzeri).
    """
    y = train[config.TARGET].to_numpy(float)
    kt, ka = te_keys(train), te_keys(apply)
    for col, key, m, stat in TE_SPECS:
        tr, ap = validation.grouped_oof_te(kt[key], y, groups_train, ka[key], m=m, stat=stat)
        if insample:
            _, tr = validation.grouped_oof_te(kt[key], y, groups_train, kt[key], m=m, stat=stat)
        train[col] = tr
        apply[col] = ap


# 30.09: PRC_KUYRUK=1 → kuyruk / pencere özellikleri (scripts/kuyruk_ozellik.py, data/kuyruk_oz.parquet) eğitim ve
# uygulama çerçevelerine eklenir; feature_list çerçevedeki tüm sütunları aldığı için bütün bileşenler bunları görür.
# Çıktı adları "_kq" içermeli (gcb_varyant / gcb_dw_tarama / d_global_cb bunu denetler).
KUYRUK = os.environ.get("PRC_KUYRUK") == "1"
GECIKME = os.environ.get("PRC_GECIKME") == "1"  # 30.09 akşam: kayan gecikme endeksi (scripts/gecikme_ozellik.py); adlar "_kqg"
GEC_COLS = ["GEC_EOBT_G15", "GEC_EOBT_G30", "GEC_SCHED_G60", "GEC_N_G15", "GEC_KENDI_FARK"]
# 01.10: PRC_GECPIST=1 → pist bazlı gecikme endeksi (scripts/gecikme_pist_ozellik.py) + varış yoğunluğu ve METAR
# (scripts/varis_wx_ozellik.py) + ileriye bakan / ortalanmış pencereler (scripts/pencere_ileri_ozellik.py) + aynı park
# yerindeki önceki / sonraki varış (scripts/park_varis_ozellik.py);
# adlar "_kqgp" (PRC_GECIKME=1 ile birlikte).
GECPIST = os.environ.get("PRC_GECPIST") == "1"
GEC_PIST_COLS = ["GEC_PIST_EOBT_G15", "GEC_PIST_EOBT_G30", "GEC_PIST_N_G15", "GEC_PIST_FARK_G15"]
VARIS_WX_COLS = ["AR_APT_15", "AR_PIST_15", "AR_PIST_PAY_60", "WX_FZ_0", "WX_FG_0", "WX_SN_0", "WX_SN_3S", "WX_TS_0",
                 "WX_GORUS", "WX_RUZGAR", "WX_HAMLE"]
ILERI_COLS = ["GECC_EOBT_15", "GECF_EOBT_15", "NMT_C15", "NMT_F15", "DEP_APT_SONRA_10", "DEP_PIST_SONRA_10", "ARR_SONRA_10",
              "GECC_PIST_15", "NMT_PIST_C15"]
PARK_VARIS_COLS = ["IN_GEC", "IN_TURN", "IN_TAMPON", "NX_GAP", "NX_T"]  # scripts/park_varis_ozellik.py (01.10)
KQ_COLS = ["KQ_TO_APT_PEN", "KQ_VARIS_PEN", "KQ_ONDE_APT", "KQ_ITME_20", "KQ_TO_PIST_PEN", "KQ_ONDE_PIST",
           "KQ_ONCEKI_PIST", "KQ_SONRAKI_PIST", "KQ_PIST_HIZ", "KQ_SERBEST"]


# 02.10 akşam: PRC_EKOZ=1 → yüzey (scripts/yuzey_ozellik.py) ve kar (scripts/kar_ozellik.py) özelliklerinden taramada kabul
# edilenler; seçim kutuk/ekoz_secim.json ({dosya: [sütunlar]}, scripts/ekoz_karar.py yazar). Adlar "_kqgpy".
EKOZ = os.environ.get("PRC_EKOZ") == "1"
# 06.10: seçim dosyası ortam değişkeniyle; final yapılandırması kutuk/ekoz_secim_final.json (pist dahil).
# Zincirler artık ekoz_secim.json'u değiştirip geri koymak zorunda değil.
EKOZ_DOSYA = os.environ.get("PRC_EKOZ_DOSYA", "ekoz_secim.json")
# 06.10: PRC_GUNSUZ=1 → MVT_DAY özellik listesinden çıkarılır (2025 takvim günü ezberi); adlar "gunsuz" içerir.
GUNSUZ = os.environ.get("PRC_GUNSUZ") == "1"


def ekoz_secim():
    import json
    return json.loads((config.ROOT / "kutuk" / EKOZ_DOSYA).read_text(encoding="utf-8"))


def kuyruk_ekle(*dfs):
    kq = pd.read_parquet(config.DATA_DIR / "kuyruk_oz.parquet", columns=["MVT_ID_mvt"] + KQ_COLS)
    for c in KQ_COLS:
        kq[c] = kq[c].astype(np.float32)
    return tuple(d.merge(kq, on="MVT_ID_mvt", how="left", validate="one_to_one") for d in dfs)


def kuyruk_denetle(ad):
    """Çıktı adı ile PRC_KUYRUK / PRC_GECIKME ayarı uyuşmalı (yeni özellikli tahmin eski dosyanın üzerine yazılmasın)."""
    assert ("_kq" in ad) == KUYRUK, f"'{ad}': ad '_kq' içeriyor = {'_kq' in ad}, PRC_KUYRUK = {KUYRUK}"
    assert ("_kqg" in ad) == GECIKME, f"'{ad}': ad '_kqg' içeriyor = {'_kqg' in ad}, PRC_GECIKME = {GECIKME}"
    assert ("_kqgp" in ad) == GECPIST, f"'{ad}': ad '_kqgp' içeriyor = {'_kqgp' in ad}, PRC_GECPIST = {GECPIST}"
    assert ("_kqgpy" in ad) == EKOZ, f"'{ad}': ad '_kqgpy' içeriyor = {'_kqgpy' in ad}, PRC_EKOZ = {EKOZ}"
    assert ("gunsuz" in ad) == GUNSUZ, f"'{ad}': ad 'gunsuz' içeriyor = {'gunsuz' in ad}, PRC_GUNSUZ = {GUNSUZ}"


def gecikme_ekle(*dfs):
    g = pd.read_parquet(config.DATA_DIR / "gecikme_oz.parquet", columns=["MVT_ID_mvt"] + GEC_COLS)
    if GECPIST:
        for f, cols in (("gecikme_pist_oz.parquet", GEC_PIST_COLS), ("varis_wx_oz.parquet", VARIS_WX_COLS),
                        ("pencere_ileri_oz.parquet", ILERI_COLS), ("park_varis_oz.parquet", PARK_VARIS_COLS)):
            g = g.merge(pd.read_parquet(config.DATA_DIR / f, columns=["MVT_ID_mvt"] + cols), on="MVT_ID_mvt", how="left",
                        validate="one_to_one")
    if EKOZ:
        for f, cols in ekoz_secim().items():
            g = g.merge(pd.read_parquet(config.DATA_DIR / f, columns=["MVT_ID_mvt"] + cols), on="MVT_ID_mvt", how="left",
                        validate="one_to_one")
    return tuple(d.merge(g, on="MVT_ID_mvt", how="left", validate="one_to_one") for d in dfs)


def prepare_split(split, insample_te=False):
    """Bölme için (eğitim, doğrulama, doğrulama ayları): eğitim satırları v126 gibi temizlenir,
    doğrulama satırları temizlenmez; TE kat içinde (ay dışı OOF) hesaplanır."""
    base = base_table("train")
    a = base[base[config.TARGET].notna()].reset_index(drop=True)
    del base
    tr_m, va_m, val_months = validation.split_masks(a["MVT_MONTH"], split)
    train = a[tr_m & clean_mask(a).to_numpy()].reset_index(drop=True)
    val = a[va_m].reset_index(drop=True)
    del a
    if KUYRUK:
        train, val = kuyruk_ekle(train, val)
    if GECIKME:
        train, val = gecikme_ekle(train, val)
    add_te(train, val, train["MVT_MONTH"].to_numpy(), insample=insample_te)
    return train, val, val_months


def prepare_full():
    """Gönderim için: 12 ayın tamamı (eğitim, v126 gibi temizlenmiş) ve testin Model A satırları (NM_TAXI_DIFF dolu;
    v126:196). TE eğitim satırlarında ay gruplu OOF, test satırlarında 12 ayın tamamından."""
    base = base_table("train")
    train = base[base[config.TARGET].notna()]
    train = train[clean_mask(train).to_numpy()].reset_index(drop=True)
    del base
    rank = base_table("rank")
    test = rank[rank["NM_TAXI_DIFF_SEC"].notna()].reset_index(drop=True)
    del rank
    if KUYRUK:
        train, test = kuyruk_ekle(train, test)
    if GECIKME:
        train, test = gecikme_ekle(train, test)
    add_te(train, test, train["MVT_MONTH"].to_numpy())
    return train, test


# ----------------------------------------------------------------------------- model matrisi
def clean_mask(df):
    """v126 eğitim temizliği (v126:65-70): TAXI ≤ 0 ve LIRF'te TAXI < 100."""
    taxi = df["TAXITIME_SEC_mvt"]
    return ~((taxi <= 0) | (df["ADEP_mvt"].eq("LIRF") & (taxi < 100)))


def feature_list(base_columns):
    """v126'nın Model A özellik listesi: birleştirme sırası (v126:50-57) ve filtre (v126:208-210).

    Sütunlarda TE zaten varsa (add_te sonrası) yeniden eklenmez.
    """
    cols = [c for c in base_columns if c not in TE_COLS]
    cols[cols.index("RWY_PHYSICAL_LENGTH_FT"):cols.index("RWY_PHYSICAL_LENGTH_FT")] = TE_COLS[:-1]  # TE v2, V3'ten önce
    cols.insert(cols.index("RWY_DISPLACED_THRESH_FT") + 1, V3_TE_COL)                           # V3 dosyasındaki yeri
    out = [c for c in cols if c not in EXCLUDE_V126 and not c.startswith("DEST_") and "holiday" not in c.lower()]
    if GUNSUZ:
        out = [c for c in out if c != "MVT_DAY"]
    assert len(out) == len(set(out)), "yinelenen özellik adı"
    return out


def matrix(df, feats, v126_bugs=True, keep_nan=False):
    """Model matrisi. v126_bugs=True: 5 kategorik dışındaki her sütun to_numeric → NaN → 0 (v126:212-215).

    keep_nan=True (v126_bugs ile): metin yine sayısala zorlanır ama eksik değer NaN kalır (LightGBM / XGBoost kendisi işler).
    """
    X = df[feats].copy()
    for c in feats:
        if c in CAT_V126:
            X[c] = X[c].fillna("nan").astype(str).astype("category")
        elif v126_bugs:
            v = pd.to_numeric(X[c], errors="coerce")
            X[c] = (v if keep_nan else v.fillna(0)).astype(np.float32)
    return X


def numeric_array(X, feats, keep_nan=False):
    """CatBoost/XGBoost girdisi (v126:259,275-276): yalnızca sayısal sütunlar, NaN → 0 (keep_nan: NaN kalır), ±1e6 kırpma."""
    num = [c for c in feats if c not in CAT_V126]
    a = X[num].to_numpy(dtype=np.float32)
    return np.clip(np.nan_to_num(a, nan=np.nan if keep_nan else 0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6), num

"""ADS-B düzeltici için ortak yardımcılar: Model A'nın dondurulan v149 harmanı (D_dw) ve ADS-B özellikleri.

D_dw ağırlıkları: kutuk/g1_agirliklar_H_C_dw_lr03_ek5_l2_10.json (xgb 0,1021; gcb6k_ort2 0,2036; e1h3 0,2211;
lgbh3 0,1236; dw_lr03_ek5_l2_10 0,3497). Bileşenler ofset (BLOCK − AOBT_3) tahmini; taksi = max(60, NM_TAXI_DIFF − ofset).
split_1 = Ocak+Temmuz 2025 doğrulama (diğer 10 ayla eğitilmiş), yani bu aylar için dışarıda bırakılmış tahminler.
"""
import json
import os

import numpy as np
import pandas as pd

from src import config

PRED = config.DATA_DIR / "preds"
ADSB_DIR = config.DATA_DIR / os.environ.get("PRC_ADSB_OZ", "adsb")  # 30.09: adsb3 = üçüncü eşleştirmeli
EPOCH_TAM = os.environ.get("PRC_EPOCH_TAM") == "1"  # 07.10: mvt_epoch 12 ayın hepsinden (bkz. mvt_epoch)
# 30.09: PRC_HARMAN ile taban harmanı seçilir. dw = v149 (varsayılan); kq = beş bileşen kuyruk özellikli (kuyruk_24.sh);
# kq4 = global CatBoost v149'dan, diğer dört bileşen kuyruk özellikli (ara harman). Ağırlıklar şimdilik D_dw'ninkiler.
HARMAN = os.environ.get("PRC_HARMAN", "dw")
HARMANLAR = {"dw": ("g1_agirliklar_H_C_dw_lr03_ek5_l2_10.json", "D_dw", "v_e1_huber1000_3t"),
             "kq": ("g1_agirliklar_kq.json", "D_kq", "v_e1_huber1000_3t_kq"),
             "kq4": ("g1_agirliklar_kq4.json", "D_kq4", "v_e1_huber1000_3t_kq"),
             "kqg": ("g1_agirliklar_kqg.json", "D_kqg", "v_e1_huber1000_3t_kqg"),
             "kqgp": ("g1_agirliklar_kqgp.json", "D_kqgp", "v_e1_huber1000_3t_kqgp"),
             "kqgpx": ("g1_agirliklar_kqgpx.json", "D_kqgpx", "v_e1_huber1000_3t_kqgp"),
             "kqgpxg": ("g1_agirliklar_kqgpxg.json", "D_kqgpxg", "v_e1_huber1000_3t_kqgp"),
             "kqgpy": ("g1_agirliklar_kqgpy.json", "D_kqgpy", "v_e1_huber1000_3t_kqgpy"),
             # 03.10 akşam: kqgpy2 = OOF kqgpy (komşu 1. tur), TEST komşu 2. turla tam eğitim (scripts/zincir_komsu2.sh)
             "kqgpy2": ("g1_agirliklar_kqgpy.json", "D_kqgpy2", "v_e1_huber1000_3t_kqgpy"),
             # 04.10: kqgpT2 = v202 kurgusu (OOF kqgpxg, p_kilit eski E1) + TEST komşu 2. tur
             "kqgpT2": ("g1_agirliklar_kqgpT.json", "D_kqgpT2", "v_e1_huber1000_3t_kqgp"),
             # 04.10: kqgpT3 = kqgpT2 + yankı geçmişi (yanki_te_oz) test tarafında (scripts/zincir_yte.sh)
             "kqgpT3": ("g1_agirliklar_kqgpT.json", "D_kqgpT3", "v_e1_huber1000_3t_kqgp"),
             # 04.10: kqgpT4 = kqgpT2 + EUROCONTROL günlük serileri (gunluk_oz) test tarafında (scripts/zincir_gun.sh)
             "kqgpT4": ("g1_agirliklar_kqgpT.json", "D_kqgpT4", "v_e1_huber1000_3t_kqgp"),
             # 04.10: kqgpT5 = komşu 1. + 2. tur + yankı geçmişi + günlük seriler (scripts/zincir_yg.sh)
             "kqgpT5": ("g1_agirliklar_kqgpT.json", "D_kqgpT5", "v_e1_huber1000_3t_kqgp"),
             # 04.10: kqgpT6 = kqgpT5 ama GCB / DW test modelleri CV erken durdurmasına uygun iterasyonla (scripts/zincir_iter.sh)
             "kqgpT6": ("g1_agirliklar_kqgpT.json", "D_kqgpT6", "v_e1_huber1000_3t_kqgp"),
             # 05.10: kqgpT7 = kqgpT5 (v208) + test tarafında GCB 4 / DW 5 tohum (scripts/zincir_tohum.sh)
             "kqgpT7": ("g1_agirliklar_kqgpT7.json", "D_kqgpT7", "v_e1_huber1000_3t_kqgp"),
             # 05.10: kqgpT8 = kqgpT7 + 6 bölmede yeniden seçilen harman ağırlıkları (scripts/harman_agirlik6.py, zincir_agirlik.sh)
             "kqgpT8": ("g1_agirliklar_kqgpT8.json", "D_kqgpT8", "v_e1_huber1000_3t_kqgp"),
             # 05.10: kqgpT9 = kqgpT5 (v208, aynı tohum / ağırlık) + pist konfigürasyonu özellikleri (scripts/zincir_y1_pist.sh)
             "kqgpT9": ("g1_agirliklar_kqgpT.json", "D_kqgpT9", "v_e1_huber1000_3t_kqgp"),
             # 05.10: final yapısı — kqgpT8b = T8 ağırlıkları + tüm çok tohumlu bileşenler (lgbh3 / E1 6, GCB 4, DW 5);
             # kqgpP = aynısı pist konfigürasyonlu bileşenlerle (zincir_final_hazirlik.sh, zincir_pist_tohum.sh)
             "kqgpT8b": ("g1_agirliklar_kqgpT8b.json", "D_kqgpT8b", "v_e1_huber1000_3t_kqgp"),
             "kqgpP": ("g1_agirliklar_kqgpP.json", "D_kqgpP", "v_e1_huber1000_3t_kqgp"),
             # 05.10: yedek finaller — T8b / P bileşenleri + eski kqgpT ağırlıkları (yeni ağırlıklar LB'de kaybederse; zincir_yedek_eski_agirlik.sh)
             "kqgpT8e": ("g1_agirliklar_kqgpT8e.json", "D_kqgpT8e", "v_e1_huber1000_3t_kqgp"),
             "kqgpPe": ("g1_agirliklar_kqgpPe.json", "D_kqgpPe", "v_e1_huber1000_3t_kqgp"),
             # 05.10 gece (denetim): kqgpPe ile aynı harman, ama LIRF yankı hedge'inin p_kilit'i final E1'den (pistli, 6 tohum;
             # komşu / yankı geçmişi / pist özellikli). Önceki tüm sürümler hedge için eski kqgp E1'ini kullanıyordu.
             "kqgpPe1": ("g1_agirliklar_kqgpPe.json", "D_kqgpPe", "v_e1_huber1000_6t_pist_kqgpy"),
             # 06.10: kqgpPe ile aynı (eski ağırlıklar, aynı tohumlar, yankı hedge'i v224'teki gibi eski E1'den),
             # test bileşenleri MVT_DAY'siz (scripts/zincir_gunsuz.sh). v224 ile farkı = yalnız MVT_DAY
             "kqgpGe": ("g1_agirliklar_kqgpGe.json", "D_kqgpGe", "v_e1_huber1000_3t_kqgp"),
             # 09.10 gece: kqgpPe'nin 5 bileşeni × (1 − w) + TabM × w
             # (OOF NN_tabm_kqgp_<bölme>, test TEST_nn_tabm_kqgpy; scripts/nn_bilesen.py model=tabm tabm_k=8 amp=1)
             "kqgpPeT": ("g1_agirliklar_kqgpPeT.json", "D_kqgpPeT", "v_e1_huber1000_3t_kqgp"),
             # 10.10 (TabM güçlendirme): aynı w = 0,40, TabM 2 tohum (42, 7) × 12 epok (OOF NN_tabm_kqgp2, test TEST_nn_tabm_kqgpy_e12)
             "kqgpPeT2": ("g1_agirliklar_kqgpPeT2.json", "D_kqgpPeT2", "v_e1_huber1000_3t_kqgp"),
             # 11.10: kqgpPeT2 ile aynı ağırlık / OOF; test TabM 5 tohum (42, 7 + 2026, 1, 2; TEST_nn_tabm_kqgpy_e12t5)
             "kqgpPeT2t5": ("g1_agirliklar_kqgpPeT2.json", "D_kqgpPeT2t5", "v_e1_huber1000_3t_kqgp"),
             # 10.10 akşam (TabM kapasite): TabM d_block 1024, 2 tohum × 12 epok (OOF NN_tabm_kqgp3, test TEST_nn_tabm_kqgpy_d1024)
             "kqgpPeT3": ("g1_agirliklar_kqgpPeT3.json", "D_kqgpPeT3", "v_e1_huber1000_3t_kqgp"),
             # 03.10: kqgpT = OOF kqgpxg'nin bileşenleri, TEST sıcaklıklı (_kqgpy) tam eğitim; yalnız LB sınaması için
             "kqgpT": ("g1_agirliklar_kqgpT.json", "D_kqgpT", "v_e1_huber1000_3t_kqgp")}  # 01.10: kqgpx = XGB 1.500 ağaç
AGIRLIK = config.ROOT / "kutuk" / HARMANLAR[HARMAN][0]
TEST_AD = HARMANLAR[HARMAN][1]  # data/preds/TEST_<TEST_AD>.parquet (g_test_harman.py çıktısı)
E1_AD = HARMANLAR[HARMAN][2]  # yankı hedge'i için p_kilit kaynağı
# 30.09: PRC_IKINCI=<ad> → ikinci aşama kalıntısı (scripts/ikinci_asama.py; OOF IKINCI_cv_<ad>, test TEST_IKINCI_<ad>) tabana
# k · rhat olarak eklenir (ofset − k · rhat). k = PRC_IKINCI_K (varsayılan 0,75).
IKINCI = os.environ.get("PRC_IKINCI")
IKINCI_K = float(os.environ.get("PRC_IKINCI_K", "0.75"))


def ikinci_test(t):
    """Test çerçevesinin harman ofsetini ve taksisini ikinci aşamayla kaydırır (PRC_IKINCI yoksa aynen döner)."""
    if not IKINCI:
        return t
    r = t[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED / f"TEST_IKINCI_{IKINCI}.parquet"), on="MVT_ID_mvt", how="left",
                                validate="one_to_one")["rhat"].fillna(0).to_numpy(float)
    t = t.copy()
    nm = t["NM_TAXI_DIFF_SEC"].to_numpy(float)
    if "harman" in t:
        t["harman"] = t["harman"] - IKINCI_K * r
        t["taksi"] = np.maximum(60.0, nm - t["harman"].to_numpy(float))
    else:
        t["taksi"] = np.maximum(60.0, t["taksi"].to_numpy(float) + IKINCI_K * r)
    return t
BILESEN = {"xgb": ("B1", "xgb"), "gcb6k_ort2": ("GCB_it6000_ort2", "global_cb"),
           "e1h3": ("ADAY_v_e1_huber1000_3t", "harman"), "lgbh3": ("ADAY_v_lgb_huber1000_3t", "harman"),
           "dw_lr03_ek5_l2_10": ("GCB_dw_lr03_ek5_l2_10", "global_cb"),
           "xgb_kq": ("ADAY_v_xgb_kq", "harman"), "gcb6k_ort2_kq": ("GCB_it6000_ort2_kq", "global_cb"),
           "e1h3_kq": ("ADAY_v_e1_huber1000_3t_kq", "harman"), "lgbh3_kq": ("ADAY_v_lgb_huber1000_3t_kq", "harman"),
           "dw_kq": ("GCB_dw_lr03_ek5_l2_10_kq", "global_cb"),
           "mlp_kq": ("NN_mlp_kq", "nn"), "tabm_kq": ("NN_tabm_kq", "nn"),
           "lgb_taksi_kq": ("ADAY_v_lgb_taksi_kq", "harman"),
           "xgb_kqg": ("ADAY_v_xgb_kqg", "harman"), "gcb6k_ort2_kqg": ("GCB_it6000_ort2_kqg", "global_cb"),
           "e1h3_kqg": ("ADAY_v_e1_huber1000_3t_kqg", "harman"), "lgbh3_kqg": ("ADAY_v_lgb_huber1000_3t_kqg", "harman"),
           "dw_kqg": ("GCB_dw_lr03_ek5_l2_10_kqg", "global_cb"),
           "xgb_kqgp": ("ADAY_v_xgb_kqgp", "harman"), "gcb6k_ort2_kqgp": ("GCB_it6000_ort2_kqgp", "global_cb"),
           "e1h3_kqgp": ("ADAY_v_e1_huber1000_3t_kqgp", "harman"), "lgbh3_kqgp": ("ADAY_v_lgb_huber1000_3t_kqgp", "harman"),
           "dw_kqgp": ("GCB_dw_lr03_ek5_l2_10_kqgp", "global_cb"),
           "xgb_kqgp_1500": ("ADAY_v_xgb_kqgp_1500", "harman"),
           "gcb12k_ort2_kqgp": ("GCB_it12000_ort2_kqgp", "global_cb"),
           "xgb_kqgpy_1500": ("ADAY_v_xgb_kqgpy_1500", "harman"), "gcb12k_ort2_kqgpy": ("GCB_it12000_ort2_kqgpy", "global_cb"),
           "e1h3_kqgpy": ("ADAY_v_e1_huber1000_3t_kqgpy", "harman"),
           "lgbh3_kqgpy": ("ADAY_v_lgb_huber1000_3t_kqgpy", "harman"), "dw_kqgpy": ("GCB_dw_lr03_ek5_l2_10_kqgpy", "global_cb"),
           "xgb_T": ("ADAY_v_xgb_kqgp_1500", "harman"), "gcb_T": ("GCB_it12000_ort2_kqgp", "global_cb"),
           "e1h3_T": ("ADAY_v_e1_huber1000_3t_kqgp", "harman"), "lgbh3_T": ("ADAY_v_lgb_huber1000_3t_kqgp", "harman"),
           "dw_T": ("GCB_dw_lr03_ek5_l2_10_kqgp", "global_cb"),
           # 05.10: çok tohumlu test bileşenlerinin OOF karşılığı _T ile aynı (tohum yalnız test tarafında)
           "gcb_T4": ("GCB_it12000_ort2_kqgp", "global_cb"), "dw_T5": ("GCB_dw_lr03_ek5_l2_10_kqgp", "global_cb"),
           "e1h3_T6": ("ADAY_v_e1_huber1000_3t_kqgp", "harman"), "lgbh3_T6": ("ADAY_v_lgb_huber1000_3t_kqgp", "harman"),
           "xgb_P": ("ADAY_v_xgb_kqgp_1500", "harman"), "gcb_P4": ("GCB_it12000_ort2_kqgp", "global_cb"),
           "e1h3_P6": ("ADAY_v_e1_huber1000_3t_kqgp", "harman"), "lgbh3_P6": ("ADAY_v_lgb_huber1000_3t_kqgp", "harman"),
           "dw_P5": ("GCB_dw_lr03_ek5_l2_10_kqgp", "global_cb"),
           # 06.10: MVT_DAY'siz test bileşenlerinin OOF karşılığı _P ile aynı (değişiklik yalnız test tarafında)
           "xgb_G": ("ADAY_v_xgb_kqgp_1500", "harman"), "gcb_G4": ("GCB_it12000_ort2_kqgp", "global_cb"),
           "e1h3_G6": ("ADAY_v_e1_huber1000_3t_kqgp", "harman"), "lgbh3_G6": ("ADAY_v_lgb_huber1000_3t_kqgp", "harman"),
           "dw_G5": ("GCB_dw_lr03_ek5_l2_10_kqgp", "global_cb"),
           # 09.10 gece: TabM (k = 8, fp16, 8 epok), kqgp özellikleri; test karşılığı final özellikleriyle (g_test_harman MAP)
           "tabm_T": ("NN_tabm_kqgp", "nn"),
           "tabm_T2": ("NN_tabm_kqgp2", "nn"),
           "tabm_T3": ("NN_tabm_kqgp3", "nn")}  # 10.10 akşam: d_block 1024, 2 tohum × 12 epok (scripts/zincir_tabm3.sh)  # 10.10: 2 tohum × 12 epok (scripts/tabm_guclendir.sh)  # scripts/nn_bilesen.py (30.09)


def bilesenler():
    return json.loads(AGIRLIK.read_text(encoding="utf-8"))["bilesenler"]


def d_dw_split(split="split_1"):
    """D_dw harmanının doğrulama tahmini: MVT_ID_mvt, ADEP_mvt, MVT_MONTH, MVT_DAY, NM_TAXI_DIFF_SEC, y (ofset), ofset_taban."""
    w = json.loads(AGIRLIK.read_text(encoding="utf-8"))
    d = pd.read_parquet(PRED / f"B1_{split}.parquet",
                        columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "MVT_DAY", "NM_TAXI_DIFF_SEC", "y"])
    d["ofset_taban"] = 0.0
    bil = []
    for ad, a in zip(w["bilesenler"], w["agirlik"]):
        f, col = BILESEN[ad]
        b = pd.read_parquet(PRED / f"{f}_{split}.parquet", columns=["MVT_ID_mvt", col])
        v = d[["MVT_ID_mvt"]].merge(b, on="MVT_ID_mvt", how="left", validate="one_to_one")[col].to_numpy(float)
        assert np.isfinite(v).all(), ad
        d["ofset_taban"] += a * v
        bil.append(v)
    d["taban_std"] = np.std(np.vstack(bil), axis=0)  # bileşen uyuşmazlığı (taban belirsizliği)
    if IKINCI:
        r = d[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED / f"IKINCI_cv_{IKINCI}.parquet"), on="MVT_ID_mvt", how="left",
                                    validate="one_to_one")["rhat"].fillna(0).to_numpy(float)
        d["ofset_taban"] = d["ofset_taban"] - IKINCI_K * r
    return d


def mvt_epoch(ids):
    """MVT_ID_mvt → kalkış anı (epoch sn), ham dosyalardan (2025 Ocak/Şubat/Temmuz/Ağustos + 2026 ranking)."""
    raw = config.PRC2_DIR / "data"
    # 07.10: 02.10'dan beri düzelticiler 12 ayla eğitiliyor ama burada yalnız 6 ay
    # okunuyordu → Mart–Mayıs / Eylül–Kasım satırlarında mvt boş, yerel / yerel_n özellikleri eğitimin yarısında boş.
    # PRC_EPOCH_TAM=1 → 12 ayın hepsi (eski çıktıların tekrar üretimi için varsayılan eski davranış).
    aylar = range(1, 13) if EPOCH_TAM else (1, 2, 6, 7, 8, 12)
    fs = [config.RANK_RAW] + [raw / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12
                                             else "training_2025-12-01_2026-01-01.parquet")
                                      for m in aylar]
    parca = []
    for f in fs:
        x = pd.read_parquet(f, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
        parca.append(x[x["MVT_ID_mvt"].isin(ids)])
    x = pd.concat(parca, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    x["mvt"] = ((x["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).astype(float)
    return x[["MVT_ID_mvt", "mvt"]]


def _stand(ids):
    """MVT_ID_mvt → (ADEP_mvt, STAND_mvt), işlenmiş eğitim ve test dosyalarından."""
    parca = [pd.read_parquet(f, columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt"]) for f in (config.DEP_TRAIN, config.DEP_RANK)]
    x = pd.concat(parca, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    return x[x["MVT_ID_mvt"].isin(ids)]


def park_tablosu(en_az=3):
    """(havalimanı, park yeri) → konum: ilk noktası durağan (< 3 kt) görülen kalkışların ilk noktalarının medyanı.
    Etiket kullanmaz (yalnızca ADS-B konumu); 2025 ve 2026 günlerinin hepsinden. Park yeri
    koordinatı veriden çıkarılıyor."""
    oz = ozellikler()
    oz = oz[(oz["ADSB_ILK_GS"] < 3) & oz["ADSB_ILK_LAT"].notna()]
    oz = oz.merge(_stand(oz["MVT_ID_mvt"]), on="MVT_ID_mvt", how="inner")
    oz = oz[oz["STAND_mvt"].notna()]
    g = oz.groupby(["ADEP_mvt", "STAND_mvt"]).agg(park_lat=("ADSB_ILK_LAT", "median"), park_lon=("ADSB_ILK_LON", "median"),
                                                  park_n=("ADSB_ILK_LAT", "size"))
    return g[g["park_n"] >= en_az].reset_index()


def park_ozellik(d, tablo):
    """d (MVT_ID_mvt ve ADS-B konum sütunları) için park yerinden uzaklıklar (m): ilk noktaya, hareket başlangıcına,
    kalkış koşusu başlangıcına. Park yeri bilinmiyorsa NaN."""
    s = d[["MVT_ID_mvt"]].merge(_stand(d["MVT_ID_mvt"]), on="MVT_ID_mvt", how="left")
    s = s.merge(tablo, on=["ADEP_mvt", "STAND_mvt"], how="left")
    la, lo = s["park_lat"].to_numpy(float), s["park_lon"].to_numpy(float)

    def m(lat, lon):
        k = np.cos(np.radians(la))
        return 111_195.0 * np.hypot(np.asarray(lat, float) - la, (np.asarray(lon, float) - lo) * k)

    return pd.DataFrame({"PARK_ILK_M": m(d["ADSB_ILK_LAT"], d["ADSB_ILK_LON"]),
                         "PARK_HAR_M": m(d["ADSB_HAR_LAT"], d["ADSB_HAR_LON"]),
                         "PARK_KOSU_M": m(d["ADSB_KOSU_LAT"], d["ADSB_KOSU_LON"]),
                         "PARK_N": s["park_n"].to_numpy(float)}, index=d.index)


def taksi(nm_diff, ofset):
    return np.maximum(60.0, np.asarray(nm_diff, float) - np.asarray(ofset, float))


def ozellikler(gunler=None):
    fs = sorted(ADSB_DIR.glob("oz_*.parquet"))
    if gunler is not None:
        fs = [f for f in fs if f.stem[3:] in set(gunler)]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

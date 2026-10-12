"""Yapılandırma: yollar ve sabitler.

Girdi klasörleri salt okunurdur; bu paket onlara hiçbir şey yazmaz.
Girdi yolları ortam değişkenleriyle değiştirilebilir (PRC2_DIR, PRC4_DIR).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # depo kökü
LOG_DIR = ROOT / "kutuk"                        # deney ve gönderim kütükleri

# Final aşaması (08.10): PRC_FINAL=1 iken test = final_ranking.parquet (Oca + Şub + Haz +
# Tem 2026). Tüm ara ürünler (özellikler, tahminler) ayrı DATA_DIR'e (data_final) yazılır; data/ hiç değişmez. Eğitim
# girdileri (2025) iki modda da aynıdır.
FINAL = os.environ.get("PRC_FINAL", "") == "1"
DATA_DIR = ROOT / ("data_final" if FINAL else "data")    # ara veri / önbellek

# Önceki iki kod tabanımızın veri klasörleri (yalnız veri okunur); varsayılan: deponun iki üst klasörü.
PRC2_DIR = Path(os.environ.get("PRC2_DIR", ROOT.parent.parent / "PRC_2"))
PRC4_DIR = Path(os.environ.get("PRC4_DIR", ROOT.parent.parent / "PRC_4"))
# Açık veri dosyaları (EUROCONTROL günlük / aylık, AIU pist tablosu, METAR); varsayılan: deponun üst klasöründe.
EK_VERI = Path(os.environ.get("PRC_EK_VERI", ROOT.parent / "extra_data"))
DEP_TRAIN = PRC2_DIR / "data_2" / "processed" / "dep_train.parquet"
if FINAL:
    # PRC_2'nin taban hattı grup istatistiklerini (park–pist / operatör medyanı, sayısı) eğitim + test birlikte hesaplıyor →
    # final testiyle eğitim tablosu da yeniden üretilir (scripts/final_taban.py), eğitim ve test aynı geçişten gelir.
    DEP_TRAIN = DATA_DIR / "dep_train.parquet"
    RANK_RAW = ROOT.parent / "data" / "final_ranking.parquet"         # ham test (organizatör, 08.10)
    DEP_RANK = DATA_DIR / "dep_rank.parquet"                          # scripts/final_taban.py
    SUBMIT_TEMPLATE = ROOT.parent / "data" / "final_submitting.parquet"
    TEST_AYLAR = (1, 2, 6, 7)
    TEST_N, TEST_A_N = 670_790, 661_713                              # final_submitting; NM'li (Model A) kalkış
else:
    RANK_RAW = PRC2_DIR / "data" / "ranking.parquet"
    DEP_RANK = PRC2_DIR / "data_2" / "processed" / "dep_rank.parquet"
    SUBMIT_TEMPLATE = PRC2_DIR / "data" / "submitting.parquet"
    TEST_AYLAR = (1, 7)
    TEST_N, TEST_A_N = 344_841, 339_551

TARGET = "NM_OFFSET_SEC"        # Model A hedefi: BLOCK − AOBT_3
SEED = 42

# Testteki Model A satırlarının ay dağılımı (dep_rank, NM_TAXI_DIFF_SEC dolu; 28.09 ölçümü).
# Tüm test satırlarında oran 152.719 / 192.122'dir; Model A için biraz farklı.
TEST_MODEL_A_N = {1: 150_266, 7: 189_285}
W_FIRST = TEST_MODEL_A_N[1] / sum(TEST_MODEL_A_N.values())     # Ocak ≈ 0,442543
W_SECOND = 1.0 - W_FIRST                                        # Temmuz ≈ 0,557457

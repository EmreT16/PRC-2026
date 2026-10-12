#!/bin/bash
# 07.10: OPDI → ADS-B düzelticisi CV'leri. Karşılık kutuk/dcv_duz_taban.log (198,03).
#  1) opdi=1: OPDI özellikleri matrise (tek değişken)
#  2) opdi=1 opdi_satir=1: + ADS-B izi olmayan ama OPDI'nin yerde gördüğü satırlar eğitilir / düzeltilir
#  3) taban yeniden (satır düzeyinde karşılaştırma için; rhat dosyası her koşuda üstüne yazıldığından)
# PRC_EPOCH_TAM verilmez (tek değişkenli karşılaştırma). rhat → data/preds/ADSB_duz_cv_<ad>.parquet. Kayıt: kutuk/zincir_opdi_cv.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "^20260729_20260808" kutuk/opdi_indir.log 2>/dev/null; do sleep 20; done
echo "indirme bitti $(date +%T)"; grep -c "MB →" kutuk/opdi_indir.log; grep "İNDİRİLEMEDİ" kutuk/opdi_indir.log
python -u scripts/opdi_ozellik.py > kutuk/opdi_ozellik.log 2>&1 || { echo "opdi_ozellik hatası"; tail -5 kutuk/opdi_ozellik.log; exit 1; }
echo "özellikler bitti $(date +%T)"; tail -16 kutuk/opdi_ozellik.log
export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
kos() {  # $1 ad, sonrası ek seçenekler
  local AD=$1; shift
  python -u scripts/adsb_duzeltici.py cv $D "$@" 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_$AD.log
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_$AD.parquet
  echo "CV $AD bitti $(date +%T)"; grep "k=1.0 " kutuk/dcv_duz_$AD.log; grep "OPDI hedef" kutuk/dcv_duz_$AD.log
}
kos opdi opdi=1
kos opdi_satir opdi=1 opdi_satir=1
kos taban2
echo "zincir bitti $(date +%T)"

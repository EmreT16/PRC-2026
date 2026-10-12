#!/bin/bash
# 09.10 akşam: H1 / G2 / G2b hiç Şubat / Haziran 2025'te sınanmadı (KAYNAK: Oca, Mar, Tem, Eki, Kas). Düzeltici _42o (parkb)
# CV'si sinama=2,6 → ADSB_duz_cv_parkb_subhaz; sonra H1 ve G2 / G2b CV'leri Şub / Haz eklenmiş KAYNAK'la (PRC_SUBHAZ=1).
# Kayıt: kutuk/zincir_subhaz_cv.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
echo "başladı $(date +%T)"
test -f data/preds/ADSB_duz_cv_parkb_subhaz.parquet || { python -u scripts/adsb_duzeltici.py cv $D parkb=1 sinama=2,6 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_parkb_subhaz.log \
  && cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_parkb_subhaz.parquet; }
echo "düzeltici CV tamam $(date +%T)"; tail -12 kutuk/dcv_duz_parkb_subhaz.log
echo "bitti $(date +%T)"

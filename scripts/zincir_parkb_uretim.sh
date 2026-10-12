#!/bin/bash
# 07.10 gece: (1) üretim düzelticisi _42o = _40o + parkb=1 (CV Oca/Tem 196,58 → 196,14); (2) doğrulama ve G2 / H1 için
# OOF: sinama=3,10 (karşılık dcv_duz_bosluk_taban 167,58) ve sinama=11,12 (karşılık dcv_duz_opdi_satir_kas).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
echo "başladı $(date +%T)"
test -f data/preds/TEST_D_kqgpPe_adsb_42o.parquet || PRC_EPOCH_TAM=1 python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_42o 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpPe_42o.log
test -f data/preds/TEST_D_kqgpPe_adsb_42o.parquet && echo "üretim _42o tamam $(date +%T)" || echo "üretim _42o YOK"
python -u scripts/adsb_duzeltici.py cv $D parkb=1 sinama=3,10 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_bosluk_parkb.log
cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_bosluk_parkb.parquet
python -u scripts/adsb_duzeltici.py cv $D parkb=1 sinama=11,12 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_parkb_kas.log
cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_parkb_kas.parquet
echo "bitti $(date +%T)"
grep "k=1.0 " kutuk/dcv_duz_bosluk_taban.log kutuk/dcv_duz_bosluk_parkb.log kutuk/dcv_duz_opdi_satir_kas.log kutuk/dcv_duz_parkb_kas.log

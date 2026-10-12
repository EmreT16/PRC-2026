#!/bin/bash
# 10.10: 07.10'da düzelticiye planlı saatler (planli=1: PL_S / PL_E / PL_L /
# PL_I ve ADS-B hareketine / tabana göre farkları) CV'de 196,58 → 196,44 (iki ay iyi) verdi, LB'de hiç ölçülmedi (v263 / v265).
# Şimdiki düzeltici (taban kqgpPeT2, LightGBM + TabM v 0,7) üzerinde yeniden CV: aynı ayarlar + planli=1, model=lgb+nn tek
# koşuda (tohum 42). Karşılık: ADSB_duz_cv_parkb_tabm2_lgbnn07 (aynı ayarların iki ayrı koşusunun 0,3 / 0,7 harmanı; 193,31).
# Önceden tanımlı kabul: Oca ve Tem ayrı ayrı iyi, ağırlıklı ≥ 0,10 iyi (scripts/duz_kiyas.py) → v349 + planli ve a52 + planli.
# final_tabm.sh denemesi bitmeden başlamaz (bellek). Kayıt: kutuk/duz_planli_cv.log (+ kutuk/dcv_duz_parkb_tabm2_lgbnn_planli.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2
until grep -q "zincir bitti\|HATA" kutuk/final_tabm.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
python -u scripts/adsb_duzeltici.py cv $D parkb=1 planli=1 model=lgb+nn nn_w=0.7 nn_epok=8 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" \
  > kutuk/dcv_duz_parkb_tabm2_lgbnn_planli.log
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "HATA: CV $(date +%T)"; tail -5 kutuk/dcv_duz_parkb_tabm2_lgbnn_planli.log; exit 1; }
cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_parkb_tabm2_lgbnn_planli.parquet
python scripts/duz_kiyas.py ADSB_duz_cv_parkb_tabm2_lgbnn07 ADSB_duz_cv_parkb_tabm2_lgbnn_planli 2>&1 | grep -v -i warn
echo "bitti $(date +%T)"

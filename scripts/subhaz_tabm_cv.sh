#!/bin/bash
# 11.10: final Şub / Haz 2026'yı da içeriyor, LB yalnız Oca / Tem.
# Düzelticideki TabM (v 0,7) ve planlı saatler Şub / Haz 2025'te (sinama=2,6) sınanır: (1) LightGBM (parkb; karşılık),
# (2) + TabM, (3) + TabM + planli (final ayarları). Taban kqgpPeT2. Karşılaştırma scripts/duz_kiyas_subhaz.py.
# Düzeltici 5 tohum zincirinden sonra (bellek / ekran kartı). Kayıt: kutuk/subhaz_tabm_cv.log (+ kutuk/dcv_duz_sh_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 PRC_EPOCH_TAM=1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1 parkb=1 sinama=2,6"
until grep -q "zincir bitti\|HATA" kutuk/zincir_duz_tohum5.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
cv() { local ad=$1; shift
  test -f data/preds/ADSB_duz_cv_T2_sh_$ad.parquet && return 0
  python -u scripts/adsb_duzeltici.py cv $D "$@" 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_sh_$ad.log
  [ ${PIPESTATUS[0]} -eq 0 ] && cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_T2_sh_$ad.parquet || echo "HATA: $ad $(date +%T)"
  echo "$ad bitti $(date +%T)"; }
cv lgb
cv lgbnn model=lgb+nn nn_w=0.7 nn_epok=8
cv lgbnn_pl planli=1 model=lgb+nn nn_w=0.7 nn_epok=8
python scripts/duz_kiyas_subhaz.py ADSB_duz_cv_T2_sh_lgb ADSB_duz_cv_T2_sh_lgbnn ADSB_duz_cv_T2_sh_lgbnn_pl 2>&1 | grep -v -i warn
python scripts/duz_kiyas_subhaz.py ADSB_duz_cv_T2_sh_lgbnn ADSB_duz_cv_T2_sh_lgbnn_pl 2>&1 | grep -v -i warn | tail -n 1
echo "zincir bitti $(date +%T)"

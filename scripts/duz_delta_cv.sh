#!/bin/bash
# 10.10 akşam: düzelticideki TabM (v353 ayarları: planli + LightGBM +
# TabM v 0,7) — TabM Huber δ 600 → (1) 1.000, (2) 2.000 (RMSE ölçütüne yakın; LightGBM δ 600 aynı). Karşılık ADSB_duz_cv_parkb_tabm2_lgbnn_planli (193,121; aynı
# tohum). Kural (her biri ayrı, scripts/duz_kiyas.py): Oca ve Tem ayrı ayrı iyi, ağırlıklı ≥ 0,10 iyi → üretim.
# Kayıt: kutuk/duz_delta_cv.log (+ kutuk/dcv_duz_kap_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
for v in "dl1000 nn_delta=1000" "dl2000 nn_delta=2000"; do
  set -- $v; ad=$1; shift
  echo "$ad başladı $(date +%T)"
  python -u scripts/adsb_duzeltici.py cv $D parkb=1 planli=1 model=lgb+nn nn_w=0.7 nn_epok=8 "$@" 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" \
    > kutuk/dcv_duz_kap_$ad.log
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "HATA: $ad $(date +%T)"; tail -5 kutuk/dcv_duz_kap_$ad.log; continue; }
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_kap_$ad.parquet
  python scripts/duz_kiyas.py ADSB_duz_cv_parkb_tabm2_lgbnn_planli ADSB_duz_cv_kap_$ad 2>&1 | grep -v -i warn | tail -5
done
echo "bitti $(date +%T)"

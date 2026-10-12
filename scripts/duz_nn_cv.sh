#!/bin/bash
# 10.10: düzeltici CV'si yalnız TabM ailesiyle (model=nn), taban kqgpPeT2,
# ayarlar dcv_duz_parkb_tabm2.log (LightGBM, 194,69) ile birebir. OOF → data/preds/ADSB_duz_cv_parkb_tabm2_nn$S.parquet;
# harman değerlendirmesi scripts/duz_nn_degerlendir.py (rhat = (1 − v) · LightGBM + v · TabM). Kayıt: kutuk/dcv_duz_parkb_tabm2_nn.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
E=${1:-8}; S=$([ "$E" = "8" ] && echo "" || echo "_e$E")  # 10.10: 8 dışı epokta dosya adına ek
echo "başladı $(date +%T)"
python -u scripts/adsb_duzeltici.py cv $D parkb=1 model=nn nn_epok=$E 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_parkb_tabm2_nn$S.log
cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_parkb_tabm2_nn$S.parquet
echo "bitti $(date +%T)"; grep "k=1.0 " kutuk/dcv_duz_parkb_tabm2.log kutuk/dcv_duz_parkb_tabm2_nn$S.log

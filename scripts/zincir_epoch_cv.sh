#!/bin/bash
# 07.10: mvt_epoch 12 ay düzeltmesinin düzeltici CV'si. Karşılığı kutuk/dcv_duz_taban.log
# (aynı ayarlar, eski 6 aylık mvt_epoch). Tek fark PRC_EPOCH_TAM=1. Kayıt: kutuk/dcv_duz_epoch.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe PRC_EPOCH_TAM=1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_epoch.log
echo "bitti $(date +%T)"

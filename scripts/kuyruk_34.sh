#!/bin/bash
# 02.10 sabah: kuyruk_33 taramasının birleşimleri + tohum gürültüsü (bagging/feature_fraction tohumları 7).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
T="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 kuyruk=1 iz5=1 seyrek=s30,s10 p:num_leaves=255 tur=800"
S="p:bagging_seed=7 p:feature_fraction_seed=7"
dene() { echo "== $1 $(date +%T)"; shift; python -u scripts/adsb_duzeltici.py cv $T "$@" 2>&1 | grep -E "^k=1.0 |süre"; }
dene taban_t7 $S
dene ff06_l2_30 p:feature_fraction=0.6 p:lambda_l2=30
dene ff06_l2_30_t7 p:feature_fraction=0.6 p:lambda_l2=30 $S
dene ff06_l2_30_d400 p:feature_fraction=0.6 p:lambda_l2=30 delta=400
dene ff06_l2_30_d400_t7 p:feature_fraction=0.6 p:lambda_l2=30 delta=400 $S
echo "bitti $(date +%T)"

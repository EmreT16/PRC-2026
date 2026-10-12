#!/bin/bash
# 02.10 sabah: tohum gürültüsü (kuyruk_34'te p:bagging_seed float hatası düzeltildi) + en iyi birleşimin ikinci tohumu.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
T="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 kuyruk=1 iz5=1 seyrek=s30,s10 p:num_leaves=255 tur=800"
dene() { echo "== $1 $(date +%T)"; shift; python -u scripts/adsb_duzeltici.py cv $T "$@" 2>&1 | grep -E "^k=1.0 |süre|Error"; }
for s in 7 11; do
  S="p:bagging_seed=$s p:feature_fraction_seed=$s"
  dene taban_t$s $S
  dene ff06_l2_30_d400_t$s p:feature_fraction=0.6 p:lambda_l2=30 delta=400 $S
  dene d400_t$s delta=400 $S
done
echo "bitti $(date +%T)"

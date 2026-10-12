#!/bin/bash
# 02.10: düzeltici δ ince taraması (δ=400 600'den iyi; 300 30.09'da eski düzenekte kötüydü). Tohum 42; en iyisi 7/11 ile doğrulanır.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
T="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 kuyruk=1 iz5=1 seyrek=s30,s10 p:num_leaves=255 tur=800"
dene() { echo "== $1 $(date +%T)"; shift; python -u scripts/adsb_duzeltici.py cv $T "$@" 2>&1 | grep -E "^k=1.0 |süre|Error"; }
for d in 250 300 350 500; do dene d$d delta=$d; done
for s in 7 11; do for d in 300 350; do dene d${d}_t$s delta=$d p:bagging_seed=$s p:feature_fraction_seed=$s; done; done
echo "bitti $(date +%T)"

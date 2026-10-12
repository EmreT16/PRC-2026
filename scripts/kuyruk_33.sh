#!/bin/bash
# 02.10 sabah: ADS-B düzeltici ayar taraması (kqgpxg tabanı, iz5 + seyrek kopyalar; ayarlar iz5/seyrek öncesinden kalma).
# Taban: havalimani=0 huber δ=600 oz=4 tip=1 ek_aylar=1 255 yaprak 800 tur kuyruk=1 iz5=1 seyrek=s30,s10 (v195 düzeltici).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
T="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 kuyruk=1 iz5=1 seyrek=s30,s10"
dene() { echo "== $1 $(date +%T)"; shift; python -u scripts/adsb_duzeltici.py cv $T "$@" 2>&1 | grep -E "^k=1.0 |süre"; }
dene taban p:num_leaves=255 tur=800
dene tur1200 p:num_leaves=255 tur=1200
dene tur1600 p:num_leaves=255 tur=1600
dene yaprak511 p:num_leaves=511 tur=800
dene mdl50 p:num_leaves=255 tur=800 p:min_data_in_leaf=50
dene mdl500 p:num_leaves=255 tur=800 p:min_data_in_leaf=500
dene delta400 p:num_leaves=255 tur=800 delta=400
dene delta900 p:num_leaves=255 tur=800 delta=900
dene ff06 p:num_leaves=255 tur=800 p:feature_fraction=0.6
dene l2_30 p:num_leaves=255 tur=800 p:lambda_l2=30
echo "bitti $(date +%T)"

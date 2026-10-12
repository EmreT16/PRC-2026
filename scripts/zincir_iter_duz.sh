#!/bin/bash
# 04.10 gece: ADS-B düzelticisinin tur eğrisi (CV, Oca/Tem ay dışı; LB yok). zincir_iter_egri.sh bitince başlar.
cd "$(dirname "$0")/.."
until grep -q "zincir bitti" kutuk/zincir_iter_egri.log 2>/dev/null; do sleep 120; done
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqgpT6 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=2000 kuyruk=1 iz5=1 seyrek=s30,s10 egri=1"
echo "başladı $(date +%T)"
python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^egri|^k=1.0 |süre|Error|error" > kutuk/iter_egri_duz.log
echo "zincir bitti $(date +%T)"

#!/bin/bash
# 05.10 gece: denetim maddelerinin CV'si. zincir_hedge_e1.sh bitince başlar.
#  P1: Model A düzelticisine durum özellikleri (durum=1) — taban ↔ durum, aynı harman (kqgpPe, OOF tarafı kqgpxg kurgusu)
#  T2: B5 düzelticisine iz5 (taksi başı + park ayracı) — taban ↔ iz5=1 (üretimdeki gibi PRC_HARMAN=kqgpxg)
# Kayıt: kutuk/zincir_denetim_cv.log, ayrıntılar kutuk/dcv_*.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -qE "zincir bitti|durdu|Error|error" kutuk/zincir_hedge_e1.log 2>/dev/null; do sleep 30; done
echo "başladı $(date +%T)"
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
python -u scripts/adsb_duzeltici.py cv $D > kutuk/dcv_duz_taban.log 2>&1; echo "düzeltici taban: $? $(date +%T)"
python -u scripts/adsb_duzeltici.py cv $D durum=1 > kutuk/dcv_duz_durum.log 2>&1; echo "düzeltici durum: $? $(date +%T)"
unset PRC_DUZ_EK
export PRC_HARMAN=kqgpxg
python -u scripts/adsb_b_duzeltici5.py cv > kutuk/dcv_b5_taban.log 2>&1; echo "B5 taban: $? $(date +%T)"
python -u scripts/adsb_b_duzeltici5.py cv iz5=1 > kutuk/dcv_b5_iz5.log 2>&1; echo "B5 iz5: $? $(date +%T)"
echo "zincir bitti $(date +%T)"

#!/bin/bash
# 30.09: kuyruk özellikli taban harmanıyla uçtan uca üretim.
# Kullanım: bash scripts/uret_harman.sh <harman: kq4|kq|kqg|kqgp> <ek> <kuyruk: 0|1> "<not>" ["<ek düzeltici ayarı>"]
#   5. argüman (01.10) düzeltici üretimine eklenir, ör. "pencere=1"; verilmezse davranış aynı.
#   1) g_test_harman → TEST_D_<harman>   2) ADS-B düzeltici (adsb4, 6 ay, 255/800, yankı β=0,25, kuyruk=<0|1>)
#   3) NM'siz satırlar v5 düzeltici (8 havalimanı)   4) g_gonderim
set -e
cd "$(dirname "$0")/.."
H=$1; EK=$2; KQ=$3; NOT=$4; DUZ=$5
export PYTHONIOENCODING=utf-8 PRC_HARMAN=$H PRC_ADSB_OZ=${URET_ADSB_OZ:-adsb4} PRC_ADSB_GEVSEK=1  # URET_ADSB_OZ=adsb5: iz5 sütunları (01.10)
AD=$(python -c "import sys; sys.path.insert(0,'.'); from src import adsb as A; print(A.TEST_AD)")
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_$H.json $AD > kutuk/g_test_$AD.log 2>&1
python -u scripts/adsb_duzeltici.py uretim havalimani=0 kayip=huber k=1 oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 \
    yanki_beta=0.25 yanki_apt=LIRF kuyruk=$KQ ek=$EK $DUZ 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_${AD}${EK}.log
python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 ad=${AD}${EK}_b5 2>&1 \
    | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_b5_uretim_${AD}${EK}.log
python scripts/g_gonderim.py ${AD}_adsb${EK} 1.0 b=${AD}${EK}_b5 "$NOT" 2>&1 | grep -E "MD5|Dosya|✗"

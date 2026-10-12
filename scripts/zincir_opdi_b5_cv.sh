#!/bin/bash
# 07.10: B5'e OPDI. Karşılık kutuk/dcv_b5_d5_12ay.log (12 ay, iz5 + nmsiz + damga5).
#  1) opdi=1  2) opdi=1 opdi_satir=1 + PRC_EPOCH_TAM=1. OPDI sütunları B çerçevelerine de ekleniyor (b_hazirla).
# zincir_opdi_kasim.sh bitince başlar. Kayıt: kutuk/zincir_opdi_b5_cv.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -qE "zincir bitti|durdu" kutuk/zincir_opdi_kasim.log 2>/dev/null; do sleep 30; done
export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6
B="iz5=1 nmsiz=1 damga5=1"
echo "başladı $(date +%T)"
python -u scripts/adsb_b_duzeltici5.py cv $B opdi=1 > kutuk/dcv_b5_12ay_opdi.log 2>&1
cp data/preds/ADSB_B5_cv.parquet data/preds/ADSB_B5_cv_opdi.parquet
echo "B5 opdi bitti $(date +%T)"; grep -A7 "izli B" kutuk/dcv_b5_12ay_opdi.log | head -7
PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py cv $B opdi=1 opdi_satir=1 > kutuk/dcv_b5_12ay_opdi_satir_ep.log 2>&1
cp data/preds/ADSB_B5_cv.parquet data/preds/ADSB_B5_cv_opdi_satir_ep.parquet
echo "B5 opdi_satir + epoch bitti $(date +%T)"; grep -A7 "izli B" kutuk/dcv_b5_12ay_opdi_satir_ep.log | head -7
echo "zincir bitti $(date +%T)"

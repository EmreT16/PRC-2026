#!/bin/bash
# 07.10 gece: B5 (iz5 + NM'siz + damga5 + 12 ay) için Huber δ 600 / 1.000 / 1.500 ve k 1 / 1,25 / 1,5 CV'si. Kayıt: kutuk/dcv_b5_12ay_d*.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6
for d in 600 1000 1500; do
  python -u scripts/adsb_b_duzeltici5.py cv iz5=1 nmsiz=1 damga5=1 delta=$d > kutuk/dcv_b5_12ay_d$d.log 2>&1
  echo "delta $d: $? $(date +%T)"
done
echo "zincir bitti $(date +%T)"

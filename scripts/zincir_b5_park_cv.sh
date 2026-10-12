#!/bin/bash
# 07.10 gece: B5 CV'si yeni Model B tabanıyla (boof=_dparkktah; park yeri / komşu taksi). Karşılık
# kutuk/dcv_b5_12ay_opdi_satir_ep.log (aynı ayarlar, eski taban). İkinci koşu: EHAM ve LSZH de yeni tabanla (v126apt=EDDF).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1
B="iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1"
echo "başladı $(date +%T)"
python -u scripts/adsb_b_duzeltici5.py cv $B boof=_dparkktah > kutuk/dcv_b5_bpark.log 2>&1
cp data/preds/ADSB_B5_cv.parquet data/preds/ADSB_B5_cv_bpark.parquet
python -u scripts/adsb_b_duzeltici5.py cv $B boof=_dparkktah v126apt=EDDF > kutuk/dcv_b5_bpark_ehlz.log 2>&1
cp data/preds/ADSB_B5_cv.parquet data/preds/ADSB_B5_cv_bpark_ehlz.parquet
echo "bitti $(date +%T)"
for f in dcv_b5_12ay_opdi_satir_ep dcv_b5_bpark dcv_b5_bpark_ehlz; do echo "== $f"; grep -E "^k=0.0 |^k=1.25 " kutuk/$f.log; done

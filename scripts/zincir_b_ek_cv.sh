#!/bin/bash
# 07.10 gece: (1) B5 + parkb (yeni B tabanı, 8 havalimanı; karşılık kutuk/dcv_b5_bpark_ehlz.log),
# (2) Model B tüm satırlı + park, ktah + yüzey, hava, varış (karşılık b_park_ktah_cv: 6 havalimanı 775,2).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
python -u scripts/b_model_cv_tum.py durum=park,ktah,yuzey,wx,varis > kutuk/b_park_ek_cv.log 2>&1
python scripts/b_park_degerlendir.py "" _dparkktah _dparkktahyuzeywxvaris >> kutuk/b_park_ek_cv.log 2>&1
echo "B tamam $(date +%T)"
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 \
  python -u scripts/adsb_b_duzeltici5.py cv iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 boof=_dparkktah v126apt=EDDF parkb=1 > kutuk/dcv_b5_bpark_ehlz_parkb.log 2>&1
cp data/preds/ADSB_B5_cv.parquet data/preds/ADSB_B5_cv_bpark_ehlz_parkb.parquet
echo "bitti $(date +%T)"
tail -16 kutuk/b_park_ek_cv.log
for f in dcv_b5_bpark_ehlz dcv_b5_bpark_ehlz_parkb; do echo "== $f"; grep -E "^k=1.0 |^k=1.25 |^k=1.5 " kutuk/$f.log; done

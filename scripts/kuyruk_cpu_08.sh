#!/bin/bash
# 08.10 sabah — işlemci kuyruğu (zincir_duz_lgbcb.sh bitince): (1) B5'e LightGBM + CatBoost (karşılık dcv_b5_bpark_ehlz.log,
# k = 1,25); (2) G2'ye CatBoost (lgb+cb ve cb; düzeltici lgbcb kabul edildiyse onun OOF'larıyla, değilse parkb).
# Kayıt: kutuk/kuyruk_cpu_08.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for i in $(seq 1 300); do grep -qE "zincir bitti|KAPI: kazanç yok|YOK|çıktısı yok" kutuk/zincir_duz_lgbcb.log && break; sleep 60; done
echo "başladı $(date +%T)"
B="iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1"
if [ ! -f data/preds/ADSB_B5_cv_lgbcb.parquet ]; then
  PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 \
    python -u scripts/adsb_b_duzeltici5.py cv $B boof=_dparkktah v126apt=EDDF model=lgb+cb 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_b5_lgbcb.log
  cp data/preds/ADSB_B5_cv.parquet data/preds/ADSB_B5_cv_lgbcb.parquet
fi
echo "B5 CV tamam $(date +%T)"
for f in dcv_b5_bpark_ehlz dcv_b5_lgbcb; do echo "== $f"; grep -E "^ +ay 1|^k=1.25 " kutuk/$f.log; done
if [ -f data/preds/TEST_D_kqgpPe_adsb_43o_ha.parquet ]; then SRC=lgbcb; AD=D_kqgpPe_adsb_43o_ha; else SRC=parkb; AD=D_kqgpPe_adsb_42o_ha; fi
echo "G2 kaynağı: $SRC ($AD)"
for M in lgb+cb cb; do
  PRC_HARMAN=kqgpPe PRC_ADSB_OZ=adsb5 PRC_DUZ_CV=$SRC PRC_G2_MODEL=$M python -u scripts/gecikme_asama2.py $AD ha=1 2>&1 | grep -v -i warn > kutuk/g2_${M/+/}_$SRC.log
  echo "G2 $M:"; grep -E "dışarıda|toplam" kutuk/g2_${M/+/}_$SRC.log
done
echo "bitti $(date +%T)"

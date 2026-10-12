#!/bin/bash
# 10.10: Model B düzelticisinde (B5) TabM ikinci aile mi? A düzelticisinde LightGBM +
# TabM (v 0,7) CV −1,44. B5 CV'si üretim ayarlarıyla (iz5, nmsiz, damga5, opdi, opdi_satir; parkb yok), taban boof=_dparkktah,
# v126apt=EDDF (karşılık kutuk/dcv_b5_bpark_ehlz.log): önce LightGBM (yeniden; aynı kod), sonra yalnız TabM (8 epok) →
# scripts/b5_nn_degerlendir.py. A düzelticisinin üretim zinciri (zincir_duz_nn_aday.sh) bitmeden başlamaz (bellek).
# Kayıt: kutuk/b5_nn_cv.log (+ kutuk/dcv_b5_lgb10.log, kutuk/dcv_b5_nn10.log, kutuk/b5_nn_degerlendir.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1
until grep -q "zincir bitti\|HATA" kutuk/zincir_duz_nn_aday.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
B="iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 boof=_dparkktah v126apt=EDDF"
test -f data/preds/ADSB_B5_cv_lgb10.parquet || python -u scripts/adsb_b_duzeltici5.py cv $B cikti=_lgb10 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_b5_lgb10.log
test -f data/preds/ADSB_B5_cv_lgb10.parquet || { echo "HATA: LightGBM CV $(date +%T)"; exit 1; }
echo "LightGBM tamam $(date +%T)"; grep -E "^k=1.0 |^k=1.25 " kutuk/dcv_b5_lgb10.log
python -u scripts/adsb_b_duzeltici5.py cv $B model=nn nn_epok=8 cikti=_nn10 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_b5_nn10.log
test -f data/preds/ADSB_B5_cv_nn10.parquet || { echo "HATA: TabM CV $(date +%T)"; exit 1; }
echo "TabM tamam $(date +%T)"; grep -E "^k=1.0 |^k=1.25 " kutuk/dcv_b5_nn10.log
python scripts/b5_nn_degerlendir.py > kutuk/b5_nn_degerlendir.log 2>&1
cat kutuk/b5_nn_degerlendir.log
echo "bitti $(date +%T)"

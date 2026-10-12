#!/bin/bash
# 10.10: B5'e planlı saatler (planli=1; B satırlarında yalnız tarife, nmsiz=1 NM'ye bağlı PL_E / PL_L / PL_I'yı atar).
# Güncel B tabanıyla (boof=_guncel, v126apt=EDDF); karşılık ADSB_B5_cv_guncel (aynı ayarlar, planlısız). Kural scripts/b5_kiyas.py.
# Kayıt: kutuk/b5_planli_cv.log (+ kutuk/dcv_b5_guncel_planli.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1
echo "başladı $(date +%T)"
python -u scripts/adsb_b_duzeltici5.py cv iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 planli=1 boof=_guncel v126apt=EDDF cikti=_guncel_planli 2>&1 \
  | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_b5_guncel_planli.log
test -f data/preds/ADSB_B5_cv_guncel_planli.parquet || { echo "HATA: B5 CV $(date +%T)"; tail -5 kutuk/dcv_b5_guncel_planli.log; exit 1; }
python scripts/b5_kiyas.py ADSB_B5_cv_guncel ADSB_B5_cv_guncel_planli 2>&1 | grep -v -i warn
echo "bitti $(date +%T)"

#!/bin/bash
# 10.10: B5 k'sının güncel B tabanıyla yeniden CV'si (scripts/b5_k_degerlendir.py başlığı). planli CV'si bitmeden başlamaz (bellek).
# Kayıt: kutuk/b5_k_cv.log (+ kutuk/dcv_b5_guncel.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1
until grep -q "^bitti\|HATA" kutuk/duz_planli_cv.log 2>/dev/null; do sleep 30; done
echo "başladı $(date +%T)"
python -u scripts/adsb_b_duzeltici5.py cv iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 boof=_guncel v126apt=EDDF cikti=_guncel 2>&1 \
  | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_b5_guncel.log
test -f data/preds/ADSB_B5_cv_guncel.parquet || { echo "HATA: B5 CV $(date +%T)"; exit 1; }
python scripts/b5_k_degerlendir.py 2>&1 | grep -v -i warn
echo "bitti $(date +%T)"

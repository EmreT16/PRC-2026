#!/bin/bash
# 10.10 gece: B5 CV'si 12 ayın B satırlarıyla (aylar=12); güncel B tabanı (boof=_guncel,
# v126apt=EDDF). (1) LightGBM karşılık (_g12), (2) + planli (_g12pl), (3) yalnız TabM 8 epok (_g12nn; B μ 5 tohum zinciri
# bittikten sonra, ekran kartı) → scripts/b5_degerlendir12.py (kural: iç içe toplam iyi, ≥ 8/12 ay, ≥ 5/8 havalimanı).
# Kayıt: kutuk/b5_12ay.log (+ kutuk/dcv_b5_g12*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1
B="iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 boof=_guncel v126apt=EDDF aylar=12"
cv() { local ek=$1; shift; test -f data/preds/ADSB_B5_cv$ek.parquet || python -u scripts/adsb_b_duzeltici5.py cv $B "$@" cikti=$ek 2>&1 \
  | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_b5$ek.log; test -f data/preds/ADSB_B5_cv$ek.parquet || echo "HATA: $ek $(date +%T)"; }
until grep -q "zincir bitti\|HATA" kutuk/final_tabm.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
cv _g12
echo "LightGBM tamam $(date +%T)"
cv _g12pl planli=1
python scripts/b5_degerlendir12.py aday=ADSB_B5_cv_g12pl 2>&1 | grep -v -i warn | tail -4
until grep -q "zincir bitti\|HATA" kutuk/zincir_b_mu_t5.log 2>/dev/null; do sleep 60; done
echo "TabM başladı $(date +%T)"
cv _g12nn model=nn nn_epok=8
python scripts/b5_degerlendir12.py nn=ADSB_B5_cv_g12nn 2>&1 | grep -v -i warn | tail -4
echo "bitti $(date +%T)"

#!/bin/bash
# 07.10: OPDI'nin EDDF kapsaması Oca/Tem 2025'te yok denecek kadar az (Ocak'ta 10 kalkış),
# Kasım–Aralık 2025'te zengin → aynı CV'ler sınama=11,12 ile (diğer 10 ayla eğit, Kas / Ara'da ölç; ağırlık eşit).
# zincir_opdi_cv.sh bitince başlar. Kayıt: kutuk/zincir_opdi_kasim.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -qE "zincir bitti|hatası" kutuk/zincir_opdi_cv.log 2>/dev/null; do sleep 30; done
grep -q "hatası" kutuk/zincir_opdi_cv.log && { echo "ana zincir hatalı; durdu"; exit 1; }
export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 sinama=11,12"
kos() {
  local AD=$1; shift
  python -u scripts/adsb_duzeltici.py cv $D "$@" 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_$AD.log
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_$AD.parquet
  echo "CV $AD bitti $(date +%T)"; grep "k=1.0 " kutuk/dcv_duz_$AD.log; grep "OPDI hedef" kutuk/dcv_duz_$AD.log
}
echo "başladı $(date +%T)"
kos taban_kas
kos opdi_kas opdi=1
kos opdi_satir_kas opdi=1 opdi_satir=1
echo "zincir bitti $(date +%T)"

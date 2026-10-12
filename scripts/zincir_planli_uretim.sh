#!/bin/bash
# 07.10 akşam: Model A düzelticisi _41o = _40o (opdi + opdi_satir + epoch) + planli=1 (CV 196,58 → 196,44). Yalnız düzeltici
# çıktısı (TEST_D_kqgpPe_adsb_41o); gönderim 08.10'da v259 / v261 sonucuna göre B tarafıyla birleştirilir.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe PRC_EPOCH_TAM=1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1 planli=1"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_41o 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpPe_41o.log
echo "bitti $(date +%T)"; grep -E "yankı|eğitim satırı" kutuk/adsb_uretim_D_kqgpPe_41o.log

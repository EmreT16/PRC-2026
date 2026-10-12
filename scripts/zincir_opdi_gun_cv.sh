#!/bin/bash
# 07.10 gece (doluluk denetimi): düzelticiye o gün o havalimanının OPDI eşleşme payı (opdi_gun=1). OPDI kaynağında
# 2–10 / 23–24 Oca 2026 yok (test satırlarının %15,7'si). Karşılık kutuk/dcv_duz_opdi_satir.log (196,58; aynı ayarlar,
# epoch yok). Kayıt: kutuk/dcv_duz_opdi_gun.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
echo "başladı $(date +%T)"
python -u scripts/adsb_duzeltici.py cv $D opdi_gun=1 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_opdi_gun.log
cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_opdi_gun.parquet
echo "bitti $(date +%T)"; grep "k=1.0 " kutuk/dcv_duz_opdi_gun.log

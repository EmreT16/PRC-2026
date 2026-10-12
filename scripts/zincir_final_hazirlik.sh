#!/bin/bash
# 05.10: pistsiz final yapısı — T8 ağırlıkları + çok tohum (lgbh3 / E1 6, GCB 4, DW 5; XGB 2)
# → düzeltici (ek=_34a) → gönderim (lobt=1) → Y1 havalimanı + pist kalibrasyonu. Kayıt: kutuk/zincir_final_hazirlik.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqgpT8b PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
echo "başladı $(date +%T)"
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpT8b.json D_kqgpT8b > kutuk/g_test_D_kqgpT8b.log 2>&1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_34a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpT8b_34a.log
python scripts/g_gonderim.py D_kqgpT8b_adsb_34a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "v211 ile aynı (T8 ağırlıkları, GCB 4 / DW 5 tohum) + lgbh3 / E1 6 tohum (1, 2, 3 eklendi) + LOBT. v211 ile farkı = LightGBM tohumları (küçük)." 2>&1 | grep -E "Dosya|MD5"
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1)
python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist katmanı + LOBT: pistsiz final yapısı (T8 ağırlıkları, tüm bileşenlerde çok tohum)." 2>&1 | tail -n 4
echo "zincir bitti $(date +%T)"

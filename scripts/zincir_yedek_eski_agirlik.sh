#!/bin/bash
# 05.10: yedek final adayları — v218 / v220 ile aynı yapı (tüm çok tohum, Y1 havalimanı + pist,
# LOBT) ama eski kqgpT ağırlıklarıyla (v208 / v212). Yeni (T8) ağırlıklar LB'de kaybederse kullanılır.
# kqgpT8e (pistsiz, ek=_36a) → kqgpPe (pistli, ek=_37a). Kayıt: kutuk/zincir_yedek_eski_agirlik.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
echo "başladı $(date +%T)"
for X in "kqgpT8e _36a pistsiz" "kqgpPe _37a pistli"; do
  set -- $X; H=$1; EK=$2; TUR=$3
  export PRC_HARMAN=$H
  python -u scripts/g_test_harman.py kutuk/g1_agirliklar_$H.json D_$H > kutuk/g_test_D_$H.log 2>&1 || { echo "harman hatası $H"; exit 1; }
  python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=$EK 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_${H}${EK}.log
  python scripts/g_gonderim.py D_${H}_adsb${EK} 1.0 b=D_kqgpxg_17a_b5 lobt=1 "Yedek $TUR final yapısı (tüm çok tohum) + ESKİ kqgpT ağırlıkları (v208 / v212) + LOBT. Yeni ağırlıklı karşılığından farkı = yalnız harman ağırlıkları." 2>&1 | grep -E "Dosya|MD5"
  SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1)
  python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist katmanı + LOBT: yedek $TUR final (eski ağırlıklar)." 2>&1 | tail -n 4
  echo "$H bitti $(date +%T)"
done
echo "zincir bitti $(date +%T)"

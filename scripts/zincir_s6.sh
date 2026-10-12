#!/bin/bash
# 02.10 gece: ADS-B düzelticisini 8 aydan 12 aya (+ Nisan, Mayıs, Eylül, Ekim 2025). zincir_s4.sh türevi;
# bekler: adsb_hat (indirme) ve kuyruk_40 (split_5/6). CV 8 ay ↔ 12 ay, ≥ 0,15 iyiyse v199 + 12 ay gönderimi (ek=_20a).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
sayi() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count"; }
while [ "$(sayi adsb_hat.py 2025-04 | tr -d '\r')" != "0" ]; do sleep 300; done
echo "indirme bitti $(date +%T); Nis/May/Eyl/Eki gün: $(ls ../PRC_adsb/noktalar | grep -cE '^2025-(0[459]|10)')"
export PRC_ADSB_GEVSEK=1 PRC_ADSB_IZ5=1
PRC_ADSB_OZ=adsb5 python -u scripts/adsb_ozellik.py hepsi isci=6 > kutuk/adsb5_oz_s56.log 2>&1
PRC_ADSB_OZ=adsb5_s30 PRC_ADSB_SEYRELT=0.3 python -u scripts/adsb_ozellik.py hepsi isci=6 > kutuk/adsb5_s30_oz_s56.log 2>&1
PRC_ADSB_OZ=adsb5_s10 PRC_ADSB_SEYRELT=0.1 python -u scripts/adsb_ozellik.py hepsi isci=6 > kutuk/adsb5_s10_oz_s56.log 2>&1
echo "özellikler bitti $(date +%T); adsb5 Nis/May/Eyl/Eki oz: $(ls data/adsb5 | grep -cE 'oz_2025-(0[459]|10)')"
unset PRC_ADSB_IZ5
until grep -q "kuyruk bitti" kutuk/kuyruk_40.log 2>/dev/null; do sleep 300; done
grep -q "islemci: 0" kutuk/s56_cpu.done && grep -q "ekran: 0" kutuk/s56_gpu.done || { echo "split_5/6 hatalı; durdu $(date +%T)"; exit 1; }
export PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
echo "== düzeltici CV, 8 ay $(date +%T)"
PRC_DUZ_EK=split_4 python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre" | tee kutuk/duz_cv_s6yok.txt
echo "== düzeltici CV, 12 ay $(date +%T)"
PRC_DUZ_EK=split_4,split_5,split_6 python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre" | tee kutuk/duz_cv_s6.txt
a=$(grep "^k=1.0" kutuk/duz_cv_s6yok.txt | awk '{print $NF}'); b=$(grep "^k=1.0" kutuk/duz_cv_s6.txt | awk '{print $NF}')
echo "Oca/Tem ağırlıklı: 8 ay $a → 12 ay $b"
if python -c "import sys; sys.exit(0 if float('$b') <= float('$a') - 0.15 else 1)"; then
  echo "== KABUL: üretim $(date +%T)"
  python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpxg.json D_kqgpxg > kutuk/g_test_D_kqgpxg_s56.log 2>&1
  PRC_DUZ_EK=split_4,split_5,split_6 python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_20a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpxg_20a.log
  python scripts/g_gonderim.py D_kqgpxg_adsb_20a 1.0 b=D_kqgpxg_17a_b5 "kqgpxg (GCB 12.000; iz5, seyreltilmiş kopyalar, düzeltici δ=600 — v198 LB: δ=400 +0,05, D1 yok); ADS-B düzelticisi 8 yerine 12 ayla eğitildi (v199 + Nisan, Mayıs, Eylül, Ekim 2025; Model A split_5/6 OOF). CV Oca/Tem: $a → $b." 2>&1 | grep -E "MD5|Dosya|✗"
  ls -t gonderim | head -2
else
  echo "RET: kazanç < 0,15"
fi
echo "zincir bitti $(date +%T)"

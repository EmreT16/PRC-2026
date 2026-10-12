#!/bin/bash
# 02.10 akşam: ADS-B düzelticisini 6 aydan 8 aya (Kasım+Mart 2025) genişletmek.
# Bekler: adsb_hat (Mart/Kasım indirme) ve kuyruk_39 (split_4 Model A). Sonra adsb5 / s30 / s10 özellikleri, düzeltici
# CV (s4'süz ↔ s4'lü, δ=600; v198 LB δ=400 tutmadı), Oca/Tem ağırlıklı RMSE ≥ 0,15 iyiyse üretim + gönderim (v197 + s4).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
sayi() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count"; }
while [ "$(sayi adsb_hat | tr -d '\r')" != "0" ]; do sleep 300; done
echo "indirme bitti $(date +%T); Mart $(ls ../PRC_adsb/noktalar | grep -c '^2025-03') gün, Kasım $(ls ../PRC_adsb/noktalar | grep -c '^2025-11') gün"
export PRC_ADSB_GEVSEK=1 PRC_ADSB_IZ5=1
PRC_ADSB_OZ=adsb5 python -u scripts/adsb_ozellik.py hepsi isci=6 > kutuk/adsb5_oz_s4.log 2>&1
PRC_ADSB_OZ=adsb5_s30 PRC_ADSB_SEYRELT=0.3 python -u scripts/adsb_ozellik.py hepsi isci=6 > kutuk/adsb5_s30_oz_s4.log 2>&1
PRC_ADSB_OZ=adsb5_s10 PRC_ADSB_SEYRELT=0.1 python -u scripts/adsb_ozellik.py hepsi isci=6 > kutuk/adsb5_s10_oz_s4.log 2>&1
echo "özellikler bitti $(date +%T); adsb5 Mart/Kasım oz: $(ls data/adsb5 | grep -c 'oz_2025-0[3]\|oz_2025-11')"
unset PRC_ADSB_IZ5
until grep -q "kuyruk bitti" kutuk/kuyruk_39.log 2>/dev/null; do sleep 300; done
grep -q "islemci: 0" kutuk/s4_cpu.done && grep -q "ekran: 0" kutuk/s4_gpu.done || { echo "split_4 hatalı; durdu $(date +%T)"; exit 1; }
export PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
echo "== düzeltici CV, s4'süz $(date +%T)"
python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre" | tee kutuk/duz_cv_s4yok.txt
echo "== düzeltici CV, s4'lü $(date +%T)"
PRC_DUZ_S4=1 python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre" | tee kutuk/duz_cv_s4.txt
a=$(grep "^k=1.0" kutuk/duz_cv_s4yok.txt | awk '{print $NF}'); b=$(grep "^k=1.0" kutuk/duz_cv_s4.txt | awk '{print $NF}')
echo "Oca/Tem ağırlıklı: s4'süz $a → s4'lü $b"
if python -c "import sys; sys.exit(0 if float('$b') <= float('$a') - 0.15 else 1)"; then
  echo "== KABUL: üretim $(date +%T)"
  python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpxg.json D_kqgpxg > kutuk/g_test_D_kqgpxg_s4.log 2>&1
  PRC_DUZ_S4=1 python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_19a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpxg_19a.log
  python scripts/g_gonderim.py D_kqgpxg_adsb_19a 1.0 b=D_kqgpxg_17a_b5 "kqgpxg (GCB 12.000; iz5, seyreltilmiş kopyalar, düzeltici δ=600 — v198 LB: δ=400 +0,05, D1 yok); ADS-B düzelticisi 6 yerine 8 ayla eğitildi (+ Kasım, Mart 2025; Model A split_4 OOF). CV Oca/Tem: $a → $b." 2>&1 | grep -E "MD5|Dosya|✗"
  ls -t gonderim | head -2
else
  echo "RET: kazanç < 0,15"
fi
echo "zincir bitti $(date +%T)"

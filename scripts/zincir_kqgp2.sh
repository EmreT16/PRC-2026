#!/bin/bash
# 01.10 04:00: zincir_kqgp.sh'ın yerini alır. kuyruk_30 (ekran kartı) erken başlatıldı (split_1 kqg'ye göre Oca −3,21,
# Tem −5,43); bu betik lgbh3_kqgp 3 bölmesinin bitmesini bekler, karar verir (RET ise kuyruk_30'u durdurur),
# kuyruk_29'u çalıştırır, ikisi bitince düzeltici CV (kuyruk=1 ve pencere=1) ve üretim (v184).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "\[split_3\] süre\|Traceback" kutuk/kqgp_lgb_split123.log 2>/dev/null; do sleep 60; done
echo "lgbh3_kqgp bitti $(date +%T)"
if ! python scripts/kqgp_karar.py; then
    echo "RET: kuyruk_30 durduruluyor $(date +%T)"
    for p in /proc/[0-9]*; do case "$(tr '\0' ' ' < $p/cmdline 2>/dev/null)" in "bash scripts/kuyruk_30.sh"*) kill ${p#/proc/};; esac; done
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'kqgp' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    exit 0
fi
echo "kuyruk_29 başladı $(date +%T)"
bash scripts/kuyruk_29.sh; echo "kuyruk_29 bitti: $? $(date +%T)"
until grep -q "kuyruk bitti" kutuk/kuyruk_30.log 2>/dev/null; do sleep 60; done
cat kutuk/kuyruk_30.log
grep -q "kuyruk bitti: 0" kutuk/kuyruk_30.log && grep -q "cikis: 0" kutuk/d_kqgp_e1.log || { echo "zincir hatalı; durdu"; exit 1; }
export PRC_HARMAN=kqgp PRC_ADSB_OZ=adsb4 PRC_ADSB_GEVSEK=1
echo "== harman ağırlıkları (3 bölme, bilgi için) $(date +%T)"
python scripts/harman3.py kqgp_3b xgb_kqgp gcb6k_ort2_kqgp e1h3_kqgp lgbh3_kqgp dw_kqgp --taban=kutuk/g1_agirliklar_kqgp.json 2>&1 | grep -v -i warn
echo "== düzeltici CV (kqgp, kuyruk=1) $(date +%T)"
python -u scripts/adsb_duzeltici.py cv havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre"
echo "== düzeltici CV (kqgp, kuyruk=1 pencere=1) $(date +%T)"
python -u scripts/adsb_duzeltici.py cv havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 pencere=1 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre"
echo "== v184 $(date +%T)"
bash scripts/uret_harman.sh kqgp _14a 1 "Model A tabanı: kuyruk + kayan gecikme endeksi + pist bazlı gecikme endeksi (GEC_PIST_*) + varış yoğunluğu (AR_*) + METAR hava bayrakları (WX_*; IEM, kamu malı) + ileriye bakan / ortalanmış pencereler (GECC_*, NMT_*, *_SONRA_10) + aynı park yerinde önceki / sonraki varış (IN_*, NX_*) 5 bileşene gömülü (kqgp; D_dw ağırlıkları). ADS-B: adsb4, düzeltici 6 ay, 255/800, kuyruk özellikleri, LIRF yankı hedge'i β=0,25. B satırları v5 düzeltici 8 havalimanı."
echo "== v185 $(date +%T)"
bash scripts/uret_harman.sh kqgp _14b 1 "v184 ile aynı; ADS-B düzelticisi de aynı 34 ek sütunu görüyor (pencere=1). v184 ile farkı düzelticideki pencere özelliklerinin LB etkisi." "pencere=1"
ls -t gonderim | head -3
echo "zincir bitti $(date +%T)"

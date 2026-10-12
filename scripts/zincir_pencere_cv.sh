#!/bin/bash
# 01.10 gece: zincir_181 bitince (kqg tabanının 6 aylık OOF'u hazır) düzelticiye pencere=1 sınaması.
# Karşılaştırma: zincir_181 kütüğündeki "düzeltici CV (kqg)" satırları (kuyruk=1, pencere yok).
cd "$(dirname "$0")/.."
bos() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"; }
until grep -q "zincir bitti\|zincir hatalı" kutuk/zincir_181.log 2>/dev/null; do sleep 60; done
until [ "$(bos | tr -d '\r')" -ge 5 ]; do sleep 60; done
echo "başladı $(date +%T), boş bellek $(bos) GB"
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqg PRC_ADSB_OZ=adsb4 PRC_ADSB_GEVSEK=1
python -u scripts/adsb_duzeltici.py cv havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 pencere=1 2>&1 \
    | grep -v -i "warn\|X\[\"tip\"\]"
echo "bitti $(date +%T)"

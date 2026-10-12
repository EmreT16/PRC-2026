#!/bin/bash
# 04.10: lgbh3 tabanlı PIST taraması bitince harman tabanlı PIST taraması (kutuk/tarama_pist_harman.log)
cd "$(dirname "$0")/.."
sayi() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count"; }
while [ "$(sayi tarama_olay | tr -d '\r')" != "0" ]; do sleep 30; done
TARAMA_BIL=HARMAN_kqgpxg PYTHONIOENCODING=utf-8 python -u scripts/tarama_olay.py ay NOISE,PIST,PISTK,PISTC > kutuk/tarama_pist_harman.log 2>&1

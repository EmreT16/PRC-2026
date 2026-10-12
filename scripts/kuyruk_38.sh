#!/bin/bash
# 02.10 akşam: aday taraması. kar kipi hemen (kar_oz hazır); ay kipi yuzey_oz bitince.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
python -u scripts/tarama_olay.py kar NOISE,KAR,KARG,KARHEP,KARW,SIC > kutuk/tarama_kar.log 2>&1 &
while pgrep -f yuzey_ozellik > /dev/null 2>&1 || ! grep -q "kalkış |" kutuk/yuzey_oz.log; do sleep 60; done
python -u scripts/tarama_olay.py ay NOISE,YZ,NMG,YZNMG,KARHEP,SIC > kutuk/tarama_ay.log 2>&1
wait
python -u scripts/tarama_olay.py kar YZ,YZNMG > kutuk/tarama_kar_yz.log 2>&1
echo bitti >> kutuk/tarama_ay.log

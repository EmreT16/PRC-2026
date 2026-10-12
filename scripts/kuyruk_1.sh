#!/bin/bash
# A2 bittikten sonra E1 ve C vekil deneylerini sırayla çalıştırır (RAM tek iş kaldırıyor).
cd "$(dirname "$0")/.."
until grep -q "^cikis:" kutuk/a2_kalibrasyon.log; do sleep 30; done
export PYTHONIOENCODING=utf-8
python -u scripts/e1_kilitli_karisim.py > kutuk/e1_kilitli_karisim.log 2>&1; echo "cikis: $?" >> kutuk/e1_kilitli_karisim.log
python -u scripts/c_ozellik_kumesi.py > kutuk/c_ozellik_kumesi.log 2>&1; echo "cikis: $?" >> kutuk/c_ozellik_kumesi.log

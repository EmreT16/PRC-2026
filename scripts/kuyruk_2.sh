#!/bin/bash
# C vekil deneyi bittikten sonra F3 vekil deneyini çalıştırır.
cd "$(dirname "$0")/.."
until [ -f kutuk/c_ozellik_kumesi.log ] && grep -q "^cikis:" kutuk/c_ozellik_kumesi.log; do sleep 30; done
export PYTHONIOENCODING=utf-8
python -u scripts/f3_arvt.py > kutuk/f3_arvt.log 2>&1; echo "cikis: $?" >> kutuk/f3_arvt.log

#!/bin/bash
# F3 bittikten sonra D2 (Huber) vekil deneyini çalıştırır.
cd "$(dirname "$0")/.."
until [ -f kutuk/f3_arvt.log ] && grep -q "^cikis:" kutuk/f3_arvt.log; do sleep 30; done
export PYTHONIOENCODING=utf-8
python -u scripts/d2_huber.py > kutuk/d2_huber.log 2>&1; echo "cikis: $?" >> kutuk/d2_huber.log

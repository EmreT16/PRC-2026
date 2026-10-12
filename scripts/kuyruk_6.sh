#!/bin/bash
# D2 (kuyruk_4'ün son adımı) bittikten sonra: global CatBoost'u daha uzun eğitim (6.000 iterasyon, lr 0,06, depth 9).
cd "$(dirname "$0")/.."
until [ -f kutuk/d2_huber.log ] && grep -q "^cikis:" kutuk/d2_huber.log; do sleep 30; done
export PYTHONIOENCODING=utf-8
python -u scripts/gcb_varyant.py it6000 6000 0.06 9 > kutuk/gcb_it6000.log 2>&1; echo "cikis: $?" >> kutuk/gcb_it6000.log

#!/bin/bash
# Uzun global CB denemesi (gcb_it6000) bitince: 12 aylık Huber LGB (δ=1000) ve daha büyük δ denemeleri.
cd "$(dirname "$0")/.."
until [ -f kutuk/gcb_it6000.log ] && grep -q "^cikis:" kutuk/gcb_it6000.log; do sleep 30; done
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run d_tam_lgb_huber1000 scripts/d_tam_egitim.py tam_lgb_huber1000
run d2_huber_b scripts/d2_huber.py 1500 2500

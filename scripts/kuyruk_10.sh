#!/bin/bash
# D2_b (δ 1500/2500) bitince: Huber'in diğer bileşenlere yayılması ve global CB varyantları, düzenekte (iki bölme).
# Tahminler harman analizi (g1_harman_genel.py) için data/preds altında.
cd "$(dirname "$0")/.."
until [ -f kutuk/d2_huber_b.log ] && grep -q "^cikis:" kutuk/d2_huber_b.log; do sleep 30; done
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run aday_v_xgb_huber1000 scripts/aday_dogrula.py v_xgb_huber1000
run aday_v_e1_huber1000 scripts/aday_dogrula.py v_e1_huber1000
run gcb_huber1000 scripts/gcb_varyant.py huber1000 3000 0.06 9 Huber:delta=1000
run gcb_it6000 scripts/gcb_varyant.py it6000 6000 0.06 9
run aday_v_lgb_huber1000_3t scripts/aday_dogrula.py v_lgb_huber1000_3t

#!/bin/bash
# Eksik değer ve kategorik varyantları (düzenek, iki bölme); kuyruk 13 (gcb_it12000) bitince başlar.
# Huber LGB tek tohum NaN korunmuş ↔ D2 δ=1000; XGB NaN korunmuş ↔ B1 xgb; global CB 3.000 it. ek 5 kategorik /
# NaN korunmuş ↔ A2 global CB (3.000 it.).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "cikis:" kutuk/gcb_it12000.log 2>/dev/null; do sleep 30; done
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run aday_v_lgb_huber1000_nan scripts/aday_dogrula.py v_lgb_huber1000_nan
run aday_v_xgb_nan scripts/aday_dogrula.py v_xgb_nan
run gcb_kat_ek5 scripts/gcb_varyant.py kat_ek5 3000 0.06 9 kat=ek5
run gcb_nan scripts/gcb_varyant.py nan 3000 0.06 9 nan=1
run k_lgb_nan scripts/karsilastir.py D2_huber_1000:pred ADAY_v_lgb_huber1000_nan:harman
run k_xgb_nan scripts/karsilastir.py xgb ADAY_v_xgb_nan:harman
run k_gcb_kat_ek5 scripts/karsilastir.py A2_global_cb:global_cb GCB_kat_ek5:global_cb
run k_gcb_nan scripts/karsilastir.py A2_global_cb:global_cb GCB_nan:global_cb

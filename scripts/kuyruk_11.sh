#!/bin/bash
# v147: harman B (XGB, global CB 6.000 it., E1 Huber, Huber LGB 3 tohum) — 12 aylık bileşenler, test harmanı, gönderim.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run d_gcb_it6000 scripts/d_global_cb.py 6000 it6000 \
 && run d_v_e1_huber1000 scripts/d_tam_egitim.py v_e1_huber1000 \
 && run d_v_lgb_huber1000_3t scripts/d_tam_egitim.py v_lgb_huber1000_3t \
 && run g_test_B_yukseltme scripts/g_test_harman.py kutuk/g1_agirliklar_B_yukseltme.json B_yukseltme \
 && run g_gonderim_B_yukseltme scripts/g_gonderim.py B_yukseltme 1.0 "Model A: sistem_1 harman B (XGB, global CB 6000 it, E1 Huber, Huber LGB 3 tohum; 12 ay). Model B: v126"

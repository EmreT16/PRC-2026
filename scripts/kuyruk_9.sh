#!/bin/bash
# 12 aylık Huber LGB bitince: Huber LGB li beş bileşenli test harmanı ve gönderim (strong-moose_v146 beklenir).
cd "$(dirname "$0")/.."
until [ -f kutuk/d_tam_lgb_huber1000.log ] && grep -q "^cikis:" kutuk/d_tam_lgb_huber1000.log; do sleep 15; done
grep -q "^cikis: 0" kutuk/d_tam_lgb_huber1000.log || exit 1
export PYTHONIOENCODING=utf-8
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_bes_huber.json bes_huber > kutuk/g_test_bes_huber.log 2>&1 \
 && python -u scripts/g_gonderim.py bes_huber 1.0 "Model A: sistem_1 bes bilesen, duz LGB yerine Huber LGB δ=1000 (XGB, global CB, E1, Huber LGB; 12 ay). Model B: v126" \
    > kutuk/g_gonderim_bes_huber.log 2>&1
echo "cikis: $?" >> kutuk/g_gonderim_bes_huber.log

#!/bin/bash
# Beş bileşenli test harmanı bitince gönderim dosyası: Model A yalnızca bu depodan (pay 1,0), Model B v126.
cd "$(dirname "$0")/.."
until [ -f kutuk/g_bes_bilesen.log ] && grep -q "^cikis:" kutuk/g_bes_bilesen.log; do sleep 30; done
grep -q "^cikis: 0" kutuk/g_bes_bilesen.log || exit 1
export PYTHONIOENCODING=utf-8
python -u scripts/g_gonderim.py bes_bilesen 1.0 "Model A: sistem_1 bes bilesen (LGB, XGB, global CB, E1; 12 ay). Model B: v126" \
  > kutuk/g_gonderim_bes_bilesen.log 2>&1; echo "cikis: $?" >> kutuk/g_gonderim_bes_bilesen.log

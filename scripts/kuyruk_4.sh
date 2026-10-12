#!/bin/bash
# F3 bittikten sonra: beş bileşenli Model A harmanının 12 aylık parçaları ve test harmanı; en son D2 (Huber).
# Gönderim yalnızca bu deponun modelleriyle kurulur (v126 ile harman yok).
cd "$(dirname "$0")/.."
until [ -f kutuk/f3_arvt.log ] && grep -q "^cikis:" kutuk/f3_arvt.log; do sleep 30; done
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run d_tam_lgb_xgb scripts/d_tam_egitim.py tam_lgb_xgb \
 && run d_global_cb scripts/d_global_cb.py \
 && run d_tam_e1_lgb42 scripts/d_tam_egitim.py tam_e1_lgb42 \
 && run g_bes_bilesen scripts/g_bes_bilesen.py
run d2_huber scripts/d2_huber.py

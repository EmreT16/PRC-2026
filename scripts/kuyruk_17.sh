#!/bin/bash
# Depthwise ikinci tur: lr 0,03 + ek 5 kategorik (± l2 10); ardından v148 harmanına (C_ort) beşinci bileşen katkısı.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run gcb_dw_tarama2 scripts/gcb_dw_tarama.py dw_lr03_ek5 dw_lr03_ek5_l2_10
for n in dw_lr03_ek5 dw_lr03_ek5_l2_10; do
  run g1_H_C_$n scripts/g1_harman_genel.py H_C_$n xgb GCB_it6000_ort2:global_cb=gcb6k_ort2 \
    ADAY_v_e1_huber1000_3t:harman=e1h3 ADAY_v_lgb_huber1000_3t:harman=lgbh3 GCB_$n:global_cb=$n
done

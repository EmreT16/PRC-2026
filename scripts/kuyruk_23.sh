#!/bin/bash
# 30.09: kuyruk / pencere özellikli (scripts/kuyruk_ozellik.py) Model A bileşenleri, split_1 + split_2 vekil sınaması.
# Karşılaştırma: aynı tanımın özelliksiz tahminleri (ADAY_v_lgb_huber1000_3t_*, ADAY_v_e1_huber1000_3t_*).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run kq_lgb_split1 scripts/aday_dogrula.py v_lgb_huber1000_3t_kq split_1 \
 && run kq_lgb_split2 scripts/aday_dogrula.py v_lgb_huber1000_3t_kq split_2 \
 && run kq_e1_split1 scripts/aday_dogrula.py v_e1_huber1000_3t_kq split_1 \
 && run kq_e1_split2 scripts/aday_dogrula.py v_e1_huber1000_3t_kq split_2

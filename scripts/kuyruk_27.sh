#!/bin/bash
# 30.09 gece: kuyruk + kayan gecikme endeksi özellikli Model A — İŞLEMCİ zinciri (lgbh3, e1h3; 3 bölme + tam). Adlar "_kqg".
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run kqg_lgb_split123 scripts/aday_dogrula.py v_lgb_huber1000_3t_kqg split_1 split_2 split_3 \
 && run kqg_e1_split123 scripts/aday_dogrula.py v_e1_huber1000_3t_kqg split_1 split_2 split_3 \
 && run d_kqg_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqg \
 && run d_kqg_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqg

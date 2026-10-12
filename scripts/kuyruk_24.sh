#!/bin/bash
# 30.09: kuyruk özellikli Model A — İŞLEMCİ zinciri (LightGBM bileşenleri: lgbh3, e1h3). kuyruk_23'ten sonra.
# Ekran kartı bileşenleri (DW, XGB, global CatBoost) kuyruk_25.sh'de paralel. Adlar "_kq" ile biter.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run kq_lgb_split3 scripts/aday_dogrula.py v_lgb_huber1000_3t_kq split_3 \
 && run kq_e1_split3 scripts/aday_dogrula.py v_e1_huber1000_3t_kq split_3 \
 && run d_kq_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kq \
 && run d_kq_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kq

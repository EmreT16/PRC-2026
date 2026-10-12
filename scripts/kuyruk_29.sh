#!/bin/bash
# 01.10 gece: kuyruk + gecikme endeksi + pist bazlı gecikme endeksi (adlar "_kqgp") — İŞLEMCİ zinciri, lgbh3 sınamasından sonra.
# lgbh3 bölmeleri zincir_kqgp içinde (sınama olarak) çalışır; burada E1 bölmeleri ve iki tam eğitim.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run kqgp_e1_split123 scripts/aday_dogrula.py v_e1_huber1000_3t_kqgp split_1 split_2 split_3 \
 && run d_kqgp_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgp \
 && run d_kqgp_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgp

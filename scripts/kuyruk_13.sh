#!/bin/bash
# v148: harman C_ort (XGB, global CB 6.000 it. iki tohum ortalaması, E1 Huber 3 tohum, Huber LGB 3 tohum) — 12 aylık
# bileşenler, test harmanı, gönderim. Ardından (bağımsız) global CB 12.000 it. doğrulaması.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run d_gcb_it6000_t2026 scripts/d_global_cb.py 6000 it6000_t2026 2026 \
 && run d_v_e1_huber1000_3t scripts/d_tam_egitim.py v_e1_huber1000_3t \
 && run gcb_ortalama_it6000_ort2 scripts/gcb_ortalama.py it6000_ort2 it6000 it6000_t2026 \
 && run g_test_C_ort scripts/g_test_harman.py kutuk/g1_agirliklar_C_ort.json C_ort \
 && run g_gonderim_C_ort scripts/g_gonderim.py C_ort 1.0 "Model A: sistem_1 harman C_ort (XGB, global CB 6000 it 2 tohum ort., E1 Huber 3 tohum, Huber LGB 3 tohum; 12 ay). Model B: v126"
run gcb_it12000 scripts/gcb_varyant.py it12000 12000 0.06 9

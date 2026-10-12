#!/bin/bash
# 01.10 akşam: kqgp'de CatBoost bileşenleri iterasyon tavanında (DW 7.999/8.000, GCB 5.975–5.999/6.000).
# DW 16.000 (gcb_dw_tarama, ayrı başlatıldı) split_2 sonucunu yazınca GCB 12.000 iterasyon split_1/2.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
until [ "$(grep -c '== split_2' kutuk/kqgp_dw_i16k.log 2>/dev/null)" -ge 2 ] || grep -q Traceback kutuk/kqgp_dw_i16k.log 2>/dev/null; do
    sleep 60
done
sleep 60
python -u scripts/gcb_varyant.py it12000_kqgp 12000 0.06 9 split_1 split_2 > kutuk/kqgp_gcb_it12000.log 2>&1
echo "bitti $? $(date +%T)" >> kutuk/kqgp_gcb_it12000.log

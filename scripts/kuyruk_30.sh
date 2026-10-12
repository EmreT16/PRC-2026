#!/bin/bash
# 01.10 gece: kuyruk + gecikme endeksi + pist bazlı gecikme endeksi (kqgp) Model A — EKRAN KARTI zinciri (DW, XGB, global CatBoost 2 tohum).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
T=dw_lr03_ek5_l2_10_kqgp
run kqgp_dw_split123 scripts/gcb_dw_tarama.py $T split_1 split_2 split_3 \
 && run kqgp_xgb_split123 scripts/aday_dogrula.py v_xgb_kqgp split_1 split_2 split_3 \
 && run d_kqgp_xgb scripts/d_tam_egitim.py v_xgb_kqgp \
 && run d_kqgp_dw scripts/d_global_cb.py 9000 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
    cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
 && run kqgp_dw_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
 && echo "DW + XGB HAZIR $(date +%T)" \
 && run kqgp_gcb_split123 scripts/gcb_varyant.py it6000_kqgp 6000 0.06 9 split_1 split_2 split_3 \
 && run kqgp_gcb_t2026_split123 scripts/gcb_varyant.py it6000_t2026_kqgp 6000 0.06 9 split_1 split_2 split_3 tohum=2026 \
 && run d_kqgp_gcb scripts/d_global_cb.py 6000 it6000_kqgp \
 && run d_kqgp_gcb_t2026 scripts/d_global_cb.py 6000 it6000_t2026_kqgp 2026 \
 && run kqgp_gcb_ort2 scripts/gcb_ortalama.py it6000_ort2_kqgp it6000_kqgp it6000_t2026_kqgp

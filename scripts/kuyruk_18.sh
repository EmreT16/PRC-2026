#!/bin/bash
# v149 (Model A'nın dondurulan sürümü): v148 harmanı + Depthwise global CB (lr 0,03, ek 5 kategorik, l2 10; 12 ay,
# 9.000 it., 3 tohum ortalaması). Ardından Model B CV'leri: B satırlı (v126_B tabanıyla) ve tüm satırlı.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
T=dw_lr03_ek5_l2_10
run d_gcb_$T scripts/d_global_cb.py 9000 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
    cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
 && run gcb_ortalama_${T}_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
 && run g_test_D_dw scripts/g_test_harman.py kutuk/g1_agirliklar_H_C_$T.json D_dw \
 && run g_gonderim_D_dw scripts/g_gonderim.py D_dw 1.0 "Model A (dondurulan): v148 harmanı + Depthwise global CB (lr 0,03, ek 5 kategorik, l2 10; 9000 it, 3 tohum ort.; 12 ay). Model B: v126"
run b_model_cv scripts/b_model_cv.py
run b_model_cv_tum scripts/b_model_cv_tum.py

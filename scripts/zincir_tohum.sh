#!/bin/bash
# 05.10 gece: v208 (kqgpT5) + test tarafında çok tohum: GCB 10.000 it.
# 2 tohum (7, 123) eklenir → 4 tohum ortalaması; DW 9.000 it. 2 tohum (123, 999) → 5 tohum. Diğer bileşenler v208'inki.
# Harman kqgpT7, düzeltici v208 ayarlarıyla (ek=_30a), gönderimde LOBT penceresi (lobt=1). Kayıt: kutuk/zincir_tohum.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
echo "başladı $(date +%T); seçim: $(cat kutuk/ekoz_secim.json | tr -d '\n ' | cut -c1-120)"
T=dw_lr03_ek5_l2_10_kqgpy
run d_th_gcb7 scripts/d_global_cb.py 10000 it12000_t7_kqgpy 7 \
&& run d_th_gcb123 scripts/d_global_cb.py 10000 it12000_t123_kqgpy 123 \
&& run th_gcb_ort4 scripts/gcb_ortalama.py it12000_ort4_kqgpy it12000_kqgpy it12000_t2026_kqgpy it12000_t7_kqgpy it12000_t123_kqgpy \
&& run d_th_dw scripts/d_global_cb.py 9000 $T 123,999 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
   cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
&& run th_dw_ort5 scripts/gcb_ortalama.py ${T}_ort5 ${T}_t42 ${T}_t2026 ${T}_t7 ${T}_t123 ${T}_t999 \
|| { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
echo "eğitim bitti $(date +%T)"
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
until grep -q "zincir bitti" kutuk/zincir_iter_duz.log 2>/dev/null; do sleep 60; done
export PRC_HARMAN=kqgpT7 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpT7.json D_kqgpT7 > kutuk/g_test_D_kqgpT7.log 2>&1
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
for ad, a, b, c in (("GCB 4 − 2 tohum", "TEST_gcb_it12000_ort2_kqgpy", "TEST_gcb_it12000_ort4_kqgpy", "global_cb"),
                    ("DW 5 − 3 tohum", "TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort3", "TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort5", "global_cb"),
                    ("harman taksi", "TEST_D_kqgpT5", "TEST_D_kqgpT7", "taksi")):
    x = pd.read_parquet(P + a + ".parquet", columns=["MVT_ID_mvt", c]).merge(pd.read_parquet(P + b + ".parquet", columns=["MVT_ID_mvt", c]), on="MVT_ID_mvt")
    f = x[c + "_y"] - x[c + "_x"]
    print(f"{ad}: RMS {np.sqrt((f ** 2).mean()):.1f}, ort {f.mean():+.2f}")
PY
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_30a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpT7_30a.log
python scripts/g_gonderim.py D_kqgpT7_adsb_30a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "v208 ile aynı (komşu + yankı geçmişi + günlük seriler; GCB 10.000 / DW 9.000 it.) ama test tarafında çok tohum: GCB 2 → 4 tohum (42, 2026, 7, 123), DW 3 → 5 tohum (42, 2026, 7, 123, 999); + LOBT penceresi (taksi [MVT − LOBT ± 3606] içine kırpılır; 2025'te kural %100). v208 ile farkı = tohum ortalaması (+ ~40 satırlık kırpma)."
ls -t gonderim | head -2
echo "zincir bitti $(date +%T)"

#!/bin/bash
# 04.10 akşam: komşu özellikleriyle CatBoost'lar CV iç erken durdurmasında çok erken doyuyor
# (GCB12k kqgpy 6 bölme: 1.623–3.927, ortanca ~3.600; DW kqgpy: 2.433–3.688), ama v202'den beri test tam eğitimi GCB 10.000 /
# DW 9.000 sabit. v208 kurgusu (komşu + yankı + günlük) üzerinde yalnız GCB (2 tohum, 4.000) ve DW (3 tohum, 3.500) yeniden;
# LGB / E1 / XGB v208'inki. v209 − v208 = iterasyon düzeltmesinin etkisi. zincir_yg.sh bitince başlar.
# Geri dönüş: TEST_*kqgpy*_yg.parquet (v208'in test dosyaları). Kayıt: kutuk/zincir_iter.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -qE "zincir bitti|durdu" kutuk/zincir_yg.log 2>/dev/null; do sleep 120; done
grep -q "zincir bitti" kutuk/zincir_yg.log || { echo "v208 zinciri hatalı; durdu $(date +%T)"; exit 1; }
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
for f in data/preds/TEST_*kqgpy*.parquet; do case "$f" in *_sicak.parquet|*_k1.parquet|*_k2.parquet|*_yte.parquet|*_gun.parquet|*_yg.parquet) ;; *) cp "$f" "${f%.parquet}_yg.parquet";; esac; done
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
echo "başladı $(date +%T); seçim: $(cat kutuk/ekoz_secim.json | tr -d '\n ')"
T=dw_lr03_ek5_l2_10_kqgpy
run d_it_dw scripts/d_global_cb.py 3500 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
   cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
&& run it_dw_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
&& run d_it_gcb scripts/d_global_cb.py 4000 it12000_kqgpy \
&& run d_it_gcb_t2026 scripts/d_global_cb.py 4000 it12000_t2026_kqgpy 2026 \
&& run it_gcb_ort2 scripts/gcb_ortalama.py it12000_ort2_kqgpy it12000_kqgpy it12000_t2026_kqgpy \
|| { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
export PRC_HARMAN=kqgpT6 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpT.json D_kqgpT6 > kutuk/g_test_D_kqgpT6.log 2>&1
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
for ad, a, b in (("GCB", "TEST_gcb_it12000_ort2_kqgpy_yg", "TEST_gcb_it12000_ort2_kqgpy"),
                 ("DW", "TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort3_yg", "TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort3"),
                 ("harman taksi", "TEST_D_kqgpT5", "TEST_D_kqgpT6")):
    c = "taksi" if "harman" in ad else "global_cb"
    x = pd.read_parquet(P + a + ".parquet", columns=["MVT_ID_mvt", c]).merge(pd.read_parquet(P + b + ".parquet", columns=["MVT_ID_mvt", c]), on="MVT_ID_mvt")
    f = x[c + "_y"] - x[c + "_x"]
    print(f"{ad}: az iterasyon − çok iterasyon RMS {np.sqrt((f ** 2).mean()):.1f}, ort {f.mean():+.1f}")
PY
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_29a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpT6_29a.log
python scripts/g_gonderim.py D_kqgpT6_adsb_29a 1.0 b=D_kqgpxg_17a_b5 "v208 ile aynı (komşu + yankı geçmişi + günlük seriler) ama GCB (2 tohum) ve DW (3 tohum) test modelleri CV iç erken durdurmasına uygun iterasyonla: GCB 10.000 → 4.000, DW 9.000 → 3.500 (komşulu modellerde erken durdurma GCB 1.623–3.927, DW 2.433–3.688). v208 ile farkı = iterasyon düzeltmesinin etkisi." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -2
echo "zincir bitti $(date +%T)"

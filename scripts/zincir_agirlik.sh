#!/bin/bash
# 05.10: v210 (kqgpT7, çok tohum) + 6 bölmede yeniden seçilen harman ağırlıkları
# (kutuk/g1_agirliklar_kqgpT8.json; kqgpy ve kqgp OOF seçimlerinin ortalaması). Düzeltici yeni ağırlıklı OOF tabanıyla
# yeniden eğitilir (adsb.py AGIRLIK aynı JSON). zincir_tohum.sh bitince başlar. Kayıt: kutuk/zincir_agirlik.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -qE "zincir bitti|durdu" kutuk/zincir_tohum.log 2>/dev/null; do sleep 60; done
grep -q "zincir bitti" kutuk/zincir_tohum.log || { echo "zincir_tohum hatalı; durdu $(date +%T)"; exit 1; }
echo "başladı $(date +%T)"
export PRC_HARMAN=kqgpT8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpT8.json D_kqgpT8 > kutuk/g_test_D_kqgpT8.log 2>&1
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
x = pd.read_parquet(P + "TEST_D_kqgpT7.parquet", columns=["MVT_ID_mvt", "taksi"]).merge(
    pd.read_parquet(P + "TEST_D_kqgpT8.parquet", columns=["MVT_ID_mvt", "taksi", "ADEP_mvt"] if False else ["MVT_ID_mvt", "taksi"]), on="MVT_ID_mvt")
f = x["taksi_y"] - x["taksi_x"]
print(f"harman taksi yeni ağırlık − eski: RMS {np.sqrt((f ** 2).mean()):.1f}, ort {f.mean():+.2f}")
PY
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
echo "== düzeltici CV (yeni ağırlıklı taban) $(date +%T)"
python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^k=0.0 |^k=1.0 " | tee kutuk/duz_cv_kqgpT8.txt
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_31a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpT8_31a.log
python scripts/g_gonderim.py D_kqgpT8_adsb_31a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "v210 ile aynı (v208 + çok tohum GCB 4 / DW 5 + LOBT penceresi) ama Model A harman ağırlıkları 6 bölmede yeniden seçildi (scripts/harman_agirlik6.py; kqgpy ve kqgp OOF seçimlerinin ortalaması: xgb 0,183 / gcb 0,183 / e1h3 0,280 / lgbh3 0,100 / dw 0,254; eski 0,102 / 0,204 / 0,221 / 0,124 / 0,350; iç içe CV kqgpy −0,17 5/6, kqgp −0,05 6/6). Düzeltici yeni ağırlıklı OOF tabanıyla. v210 ile farkı = ağırlıklar."
ls -t gonderim | head -2
echo "zincir bitti $(date +%T)"

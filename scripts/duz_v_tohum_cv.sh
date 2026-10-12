#!/bin/bash
# 11.10: düzelticide TabM ağırlığı v = 0,7 tek tohumlu TabM OOF'uyla
# seçildi; üretim 3 (5) tohum ortalıyor → ortalamalı TabM'de en iyi v kayar mı? TabM-only CV tohum 7 (cv_tohum=7; planlısız,
# v ızgarasının eski dosyalarıyla aynı düzen) → rn = ort(tohum 42, 7) → scripts/duz_nn_degerlendir.py.
# Önceden tanımlı kural: v* ≠ 0,7, v*'de ağırlıklı ≥ 0,05 iyi ve iki ayda da v = 0,7'den iyi → v değişir; yoksa 0,7 kalır.
# Şub / Haz zincirinden sonra. Kayıt: kutuk/duz_v_tohum_cv.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 \
  PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 PRC_EPOCH_TAM=1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
until grep -q "zincir bitti" kutuk/subhaz_tabm_cv.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
test -f data/preds/ADSB_duz_cv_parkb_tabm2_nn_s7.parquet || {
  python -u scripts/adsb_duzeltici.py cv $D parkb=1 model=nn nn_epok=8 cv_tohum=7 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/dcv_duz_parkb_tabm2_nn_s7.log
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "HATA: CV $(date +%T)"; exit 1; }
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_parkb_tabm2_nn_s7.parquet; }
python - <<'PY'
import numpy as np, pandas as pd
P = "data/preds/"
a = pd.read_parquet(P + "ADSB_duz_cv_parkb_tabm2_nn.parquet"); b = pd.read_parquet(P + "ADSB_duz_cv_parkb_tabm2_nn_s7.parquet")
x = a[["MVT_ID_mvt", "rhat"]].merge(b[["MVT_ID_mvt", "rhat"]], on="MVT_ID_mvt", suffixes=("", "7"), validate="one_to_one")
m = (x.rhat.abs() > 0) | (x.rhat7.abs() > 0)
print(f"TabM rhat tohum 42 ↔ 7 (izli): korelasyon {np.corrcoef(x.rhat[m], x.rhat7[m])[0, 1]:.4f}, fark RMS {np.sqrt(((x.rhat - x.rhat7)[m] ** 2).mean()):.2f}")
x["rhat"] = 0.5 * x.rhat + 0.5 * x.rhat7
x[["MVT_ID_mvt", "rhat"]].to_parquet(P + "ADSB_duz_cv_parkb_tabm2_nn_ort2.parquet", index=False)
PY
echo "--- tek tohum (42)"; python scripts/duz_nn_degerlendir.py nn=ADSB_duz_cv_parkb_tabm2_nn 2>&1 | grep -v -i warn
echo "--- iki tohum ortalaması"; python scripts/duz_nn_degerlendir.py nn=ADSB_duz_cv_parkb_tabm2_nn_ort2 2>&1 | grep -v -i warn
echo "zincir bitti $(date +%T)"

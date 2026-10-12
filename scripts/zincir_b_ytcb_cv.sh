#!/bin/bash
# 08.10: Model B yer tutucu olasılığı (yt) için CatBoost (GPU) — μ'deki LightGBM + CatBoost kazancının sınıflandırıcıdaki
# karşılığı. Özellikler v297'nin yt'si gibi park + ktah. Değerlendirme v297 düzeninde. Kayıt: kutuk/zincir_b_ytcb_cv.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
test -f data/preds/BM_cv_tum_dparkktah_ytcb_oof.parquet || python -u scripts/b_model_cv_tum.py durum=park,ktah yt_cb=1 2>&1 | grep -v -i warn > kutuk/b_ytcb_cv.log
echo "CV tamam $(date +%T)"
python scripts/b_yt_harman_degerlendir.py _dparkktah_ytcb 2>&1 | grep -v -i warn
echo "bitti $(date +%T)"

#!/bin/bash
# 08.10: Model B μ için CatBoost bileşeni (LightGBM μ ile harman; Model A'daki çeşitlilik B'de yok). CV durum grupları v295'in
# μ'sündeki gibi (park,ktah,kom,gs,pc,yte); değerlendirme melez düzende (p24 / yt / d park + ktah). Kayıt: kutuk/zincir_b_mucb_cv.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
test -f data/preds/BM_cv_tum_dparkktahkomgspcyte_cb_oof.parquet || python -u scripts/b_model_cv_tum.py durum=park,ktah,kom,gs,pc,yte mu_cb=1 2>&1 | grep -v -i warn > kutuk/b_mucb_cv.log
echo "CV tamam $(date +%T)"
python scripts/b_mu_harman_degerlendir.py _dparkktahkomgspcyte_cb 2>&1 | grep -v -i warn
echo "bitti $(date +%T)"

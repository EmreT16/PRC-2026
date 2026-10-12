#!/bin/bash
# 07.10 gece: Model B tüm satırlı model (durum=park,ktah) için 3 tohum ortalaması CV'si (fikir 3); karşılaştırma tek tohum
# _dparkktah (775,2). Kayıt: kutuk/zincir_b_tohum_cv.log, kutuk/b_tohum_cv.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
test -f data/preds/BM_cv_tum_dparkktah_t3_oof.parquet || python -u scripts/b_model_cv_tum.py durum=park,ktah tohum=3 > kutuk/b_tohum_cv.log 2>&1
echo "CV tamam $(date +%T)"
python scripts/b_park_degerlendir.py "" _dparkktah _dparkktah_t3
echo "bitti $(date +%T)"

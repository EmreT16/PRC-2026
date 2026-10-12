#!/bin/bash
# 07.10 gece: Model B (tüm satırlı) + park yeri dizisi / komşu taksi özellikleri, 4 çeyrekli CV. Karşılık kutuk/b_model_cv_tum.md.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
python -u scripts/b_model_cv_tum.py durum=park > kutuk/b_park_cv.log 2>&1
python -u scripts/b_model_cv_tum.py durum=park,ktah > kutuk/b_park_ktah_cv.log 2>&1
echo "bitti $(date +%T)"

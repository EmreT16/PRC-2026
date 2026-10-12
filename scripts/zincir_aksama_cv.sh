#!/bin/bash
# 09.10: Model B'ye aksama günü grubu (ak) — v297 / v316 μ düzeninde CV (durum park,ktah,kom,gs,pc,yte + ak, μ LightGBM + CatBoost),
# değerlendirme melez düzende (b_mu_harman_degerlendir.py; taban _dparkktahkomgspcyte_cb). Kayıt: kutuk/zincir_aksama_cv.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
python -u scripts/b_model_cv_tum.py durum=park,ktah,kom,gs,pc,yte,ak mu_cb=1 2>&1 | grep -v -i warn > kutuk/b_aksama_cv.log
echo "CV tamam $(date +%T)"
echo "== taban"; python scripts/b_mu_harman_degerlendir.py _dparkktahkomgspcyte_cb 2>&1 | grep -v -i warn
echo "== + aksama"; python scripts/b_mu_harman_degerlendir.py _dparkktahkomgspcyteak_cb 2>&1 | grep -v -i warn
echo "bitti $(date +%T)"

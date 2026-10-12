#!/bin/bash
# 08.10 sabah — ekran kartı kuyruğu (zincir_b_ytcb_cv.sh bitince): (1) Model B p24 için CatBoost CV (park + ktah;
# v297 düzeninde değerlendirme, 24 sa dahil / hariç); (2) CatBoost μ + LightGBM μ 3 tohumla üretim (sigorta; CV ölçemiyor).
# Kayıt: kutuk/kuyruk_gpu_08.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for i in $(seq 1 240); do grep -q "^bitti" kutuk/zincir_b_ytcb_cv.log && break; sleep 60; done
echo "başladı $(date +%T)"
test -f data/preds/BM_cv_tum_dparkktah_p24cb_oof.parquet || python -u scripts/b_model_cv_tum.py durum=park,ktah p24_cb=1 2>&1 | grep -v -i warn > kutuk/b_p24cb_cv.log
python scripts/b_cb_bilesen_degerlendir.py _dparkktah_p24cb p24 2>&1 | grep -v -i warn
echo "p24 CV tamam $(date +%T)"
test -f data/preds/TEST_B_tum_dpdcb3ham.parquet || python -u scripts/b_uretim_tum.py durum=park,ktah,kom,gs,pc,yte tohum=3 mu_cb=1 cb_tohum=3 ad=tum_dpdcb3ham 2>&1 | grep -v -i warn > kutuk/b_uretim_dpdcb3ham.log
test -f data/preds/TEST_B_tum_dpdcb3ham.parquet && echo "μ 3 tohum üretimi tamam $(date +%T)" || echo "μ 3 tohum üretimi YOK"
echo "bitti $(date +%T)"

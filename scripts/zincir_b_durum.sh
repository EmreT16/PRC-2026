#!/bin/bash
# 05.10 akşam: Model B tüm satırlı modele "durum" özellikleri — 4 çeyrekli CV.
# Önce taban (aynı kod, durum yok), sonra dört grup birlikte; iyi çıkarsa tek tek. Kayıt: kutuk/zincir_b_durum.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
python -u scripts/b_model_cv_tum.py > kutuk/b_durum_taban.log 2>&1; echo "taban: $? $(date +%T)"
python -u scripts/b_model_cv_tum.py durum=kom,gs,pc,yte > kutuk/b_durum_hepsi.log 2>&1; echo "hepsi: $? $(date +%T)"
echo "zincir bitti $(date +%T)"

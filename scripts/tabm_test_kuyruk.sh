#!/bin/bash
# 09.10 gece: TabM OOF'u (bölme 2–6) bitince test tahminleri, final özellik
# kümesiyle (öteki test bileşenleri gibi: PRC_EKOZ_DOSYA=ekoz_secim_final.json), 2 tohum (42, 7), k = 8, fp16, 8 epok.
#   1) normal kip (Oca / Tem LB testi)  → data/preds/TEST_nn_tabm_kqgpy.parquet
#   2) final kip (4 ay)                → data_final/preds/TEST_nn_tabm_kqgpy.parquet
# Kayıt: kutuk/tabm_test.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_final.json
until grep -q "cikis" kutuk/tabm_oof.log 2>/dev/null; do sleep 60; done
echo "normal kip başladı $(date +%T)"
python -u scripts/nn_bilesen.py tabm_kqgpy tam model=tabm epok=8 tabm_k=8 amp=1 tohum=42,7 || { echo "HATA normal $(date +%T)"; exit 1; }
echo "final kip başladı $(date +%T)"
PRC_FINAL=1 python -u scripts/nn_bilesen.py tabm_kqgpy tam model=tabm epok=8 tabm_k=8 amp=1 tohum=42,7 || { echo "HATA final $(date +%T)"; exit 1; }
echo "bitti $(date +%T)"

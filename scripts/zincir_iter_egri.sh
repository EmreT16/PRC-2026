#!/bin/bash
# 04.10 gece: sabit iterasyonların ay dışı CV eğrisiyle yeniden kontrolü (LB yok).
# İşlemci: lgbh3 + E1; ekran kartı: XGB. 3 bölme (Oca+Tem, Şub+Ağu, Haz+Ara). Kayıt: kutuk/iter_egri_{lgb,xgb}.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
echo "başladı $(date +%T)"
python -u scripts/iter_egrisi.py lgb split_1 split_2 split_3 > kutuk/iter_egri_lgb.log 2>&1 &
python -u scripts/iter_egrisi.py xgb split_1 split_2 split_3 > kutuk/iter_egri_xgb.log 2>&1 &
wait
echo "zincir bitti $(date +%T)"

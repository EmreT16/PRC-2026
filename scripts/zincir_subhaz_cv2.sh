#!/bin/bash
# 09.10 akşam: zincir_subhaz_cv.sh'nin devamı — H1 ve G2 / G2b ay dışarıda CV'leri 7 ayla (Oca, Şub, Mar, Haz, Tem, Eki, Kas).
# Üretim dosyalarına yazmamak için aday kopyası _SH (TEST_D_kqgpPe_adsb_42o_SH) ve ayrı json adı (PRC_DUZ_CV=parkbsh).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe PRC_DUZ_CV=parkbsh
until grep -q "^bitti" kutuk/zincir_subhaz_cv.log; do sleep 30; done
test -f data/preds/ADSB_duz_cv_parkb_subhaz.parquet || { echo "HATA: düzeltici CV çıktısı yok"; exit 1; }
echo "== H1 (7 ay) $(date +%T)"; python -u scripts/havalimani_agirlik.py D_kqgpPe_adsb_42o_SH 2>&1 | grep -v -i warn
echo "== G2 LightGBM $(date +%T)"; python -u scripts/gecikme_asama2.py D_kqgpPe_adsb_42o_SH_ha ha=1 2>&1 | grep -v -i warn | grep -v "^  [A-Z]\{4\}:"
echo "== G2 LightGBM + CatBoost $(date +%T)"; PRC_G2_MODEL=lgb+cb python -u scripts/gecikme_asama2.py D_kqgpPe_adsb_42o_SH_ha ha=1 2>&1 | grep -v -i warn | grep -v "^  [A-Z]\{4\}:"
echo "== G2b LightGBM + CatBoost $(date +%T)"; PRC_G2_MODEL=lgb+cb PRC_G2_TERS=1 python -u scripts/gecikme_asama2.py D_kqgpPe_adsb_42o_SH_ha ha=1 2>&1 | grep -v -i warn | grep -v "^  [A-Z]\{4\}:"
echo "bitti $(date +%T)"

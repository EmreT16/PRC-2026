#!/bin/bash
# 10.10: Model A harmanına üçüncü tür çeşitlilik — doğrudan taksi hedefli TabM
# (girdi: kqgp özellikleri + NM farkı; çıktı ofset birimine çevrilir). Pilot: split_1 (Oca / Tem 2025), tek tohum, 12 epok.
# Değerlendirme: taban = kqgpPeT2 harman OOF'u (TabM'li), düzeltici rhat = 0,3 · LightGBM + 0,7 · TabM (CV OOF'ları; yeniden
# eğitilmeden, izsiz satırlarda kesin). Önceden tanımlı ölçüt (12 ay OOF'a geçmek için): düzeltici sonrası çapraz seçimde iki
# ayda da iyi ve en iyi w'de ağırlıklı ≥ 0,10. düz_nn_e12 kuyruğu bitmeden başlamaz (ekran kartı / bellek).
# Kayıt: kutuk/tabm_tx_pilot.log (+ kutuk/tx_split1.log, kutuk/tx_degerlendir.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0
until grep -q "^bitti\|HATA" kutuk/duz_nn_e12_kuyruk.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 python -u scripts/nn_bilesen.py tabm_kqgp_tx split_1 model=tabm tabm_k=8 amp=1 epok=12 tohum=42 hedef=taksi \
  > kutuk/tx_split1.log 2>&1 || { echo "HATA: TabM taksi $(date +%T)"; tail -5 kutuk/tx_split1.log; exit 1; }
grep "^\[split" kutuk/tx_split1.log
PRC_HARMAN=kqgpPeT2 TD_RHAT=ADSB_duz_cv_parkb_tabm2_lgbnn07 python scripts/tabm_degerlendir.py tabm_kqgp_tx split_1 2>&1 | grep -v -i warn > kutuk/tx_degerlendir.log
cat kutuk/tx_degerlendir.log
echo "bitti $(date +%T)"

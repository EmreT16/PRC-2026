#!/bin/bash
# 01.10 gece: pist bazlı gecikme endeksi + varış yoğunluğu + METAR + ileriye bakan pencereler + park yeri önceki/sonraki varış (kqgp).
# 1) İşlemci zinciri (kuyruk_27) bitince lgbh3_kqgp 3 bölme.  2) Karar (kqgp_karar.py).
# 3) KABUL ise: kuyruk_29 (işlemci) hemen; kuyruk_30 (ekran kartı) kuyruk_28 ve zincir_181 bitince (bellek için).
# 4) İkisi bitince kqgp harmanıyla düzeltici CV'si ve üretim (bir sonraki strong-moose sürümü).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "kuyruk bitti" kutuk/kuyruk_27.log 2>/dev/null && [ -s data/gecikme_pist_oz.parquet ] && [ -s data/varis_wx_oz.parquet ] && grep -q "kalkış |" kutuk/oz_pencere_ileri.log 2>/dev/null && [ -s data/park_varis_oz.parquet ]; do sleep 60; done
echo "lgbh3_kqgp başladı $(date +%T)"
PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 python -u scripts/aday_dogrula.py v_lgb_huber1000_3t_kqgp split_1 split_2 split_3 \
    > kutuk/kqgp_lgb_split123.log 2>&1 || { echo "lgbh3_kqgp hata"; exit 1; }
echo "lgbh3_kqgp bitti $(date +%T)"
python scripts/kqgp_karar.py || { echo "RET: pist endeksi alınmadı; zincir durdu $(date +%T)"; exit 0; }
(until grep -q "kuyruk bitti" kutuk/kuyruk_28.log 2>/dev/null && grep -q "zincir bitti\|zincir hatalı" kutuk/zincir_181.log 2>/dev/null; do sleep 60; done
 echo "kuyruk_30 başladı $(date +%T)"; bash scripts/kuyruk_30.sh; echo "kuyruk bitti: $? $(date +%T)") > kutuk/kuyruk_30.log 2>&1 &
echo "kuyruk_29 başladı $(date +%T)"
bash scripts/kuyruk_29.sh; echo "kuyruk_29 bitti: $? $(date +%T)"
until grep -q "kuyruk bitti" kutuk/kuyruk_30.log 2>/dev/null; do sleep 60; done
cat kutuk/kuyruk_30.log
grep -q "kuyruk bitti: 0" kutuk/kuyruk_30.log && grep -q "cikis: 0" kutuk/d_kqgp_e1.log || { echo "zincir hatalı; durdu"; exit 1; }
export PRC_HARMAN=kqgp PRC_ADSB_OZ=adsb4 PRC_ADSB_GEVSEK=1
echo "== harman ağırlıkları (3 bölme, bilgi için) $(date +%T)"
python scripts/harman3.py kqgp_3b xgb_kqgp gcb6k_ort2_kqgp e1h3_kqgp lgbh3_kqgp dw_kqgp --taban=kutuk/g1_agirliklar_kqgp.json 2>&1 | grep -v -i warn
echo "== düzeltici CV (kqgp) $(date +%T)"
python -u scripts/adsb_duzeltici.py cv havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre"
echo "== üretim $(date +%T)"
bash scripts/uret_harman.sh kqgp _14a 1 "Model A tabanı: kuyruk + kayan gecikme endeksi + pist bazlı gecikme endeksi (GEC_PIST_*) + varış yoğunluğu (AR_*) + METAR hava bayrakları (WX_*; IEM, kamu malı) + ileriye bakan / ortalanmış pencereler (GECC_*, NMT_*, *_SONRA_10) + aynı park yerinde önceki / sonraki varış (IN_*, NX_*) 5 bileşene gömülü (kqgp; D_dw ağırlıkları). ADS-B: adsb4, düzeltici 6 ay, 255/800, kuyruk özellikleri, LIRF yankı hedge'i β=0,25. B satırları v5 düzeltici 8 havalimanı."
ls -t gonderim | head -2
echo "zincir bitti $(date +%T)"

#!/bin/bash
# Gece zincirleri (kuyruk_27, kuyruk_28) bitince: kqg harmanı → CV → v181 (k=1), v182 (k=1,15), v183 (kapsamasız).
cd "$(dirname "$0")/.."
until grep -q "kuyruk bitti" kutuk/kuyruk_27.log 2>/dev/null && grep -q "kuyruk bitti" kutuk/kuyruk_28.log 2>/dev/null; do sleep 60; done
cat kutuk/kuyruk_27.log kutuk/kuyruk_28.log
if ! grep -q "kuyruk bitti: 0" kutuk/kuyruk_27.log || ! grep -q "kuyruk bitti: 0" kutuk/kuyruk_28.log; then echo "zincir hatalı; durdu"; exit 1; fi
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqg PRC_ADSB_OZ=adsb4 PRC_ADSB_GEVSEK=1
echo "== harman ağırlıkları (3 bölme, bilgi için) $(date +%T)"
python scripts/harman3.py kqg_3b xgb_kqg gcb6k_ort2_kqg e1h3_kqg lgbh3_kqg dw_kqg --taban=kutuk/g1_agirliklar_kqg.json 2>&1 | grep -v -i warn
echo "== düzeltici CV (kqg) $(date +%T)"
python -u scripts/adsb_duzeltici.py cv havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre"
echo "== v181 $(date +%T)"
bash scripts/uret_harman.sh kqg _13a 1 "Model A tabanı: kuyruk + kayan gecikme endeksi özellikleri 5 bileşene gömülü (kqg: xgb, gcb6k_ort2, e1h3, lgbh3, DW; D_dw ağırlıkları). Bileşen düzeyinde kuyruk özellikliye göre ayda −1,5 … −6,4 sn. ADS-B: üçüncü eşleştirme + NM'siz satırlarda uçuş numarası (adsb4); düzeltici 6 ay, 255/800, kuyruk özellikleri, LIRF yankı hedge'i β=0,25. B satırları v5 düzeltici 8 havalimanı. İkinci aşama yok (endeks modelin içinde)."
python - <<'PY'
import pandas as pd, numpy as np
g = pd.read_parquet("data/preds/TEST_D_kqg_adsb_13a.parquet")
temel = np.maximum(60.0, g["taksi_taban"].to_numpy() + g["rhat"].to_numpy())
hedge = g["taksi"].to_numpy() - temel
g["taksi"] = np.maximum(60.0, g["taksi_taban"] + 1.15 * g["rhat"]) + hedge
g.to_parquet("data/preds/TEST_D_kqg_adsb_13a_k115.parquet", index=False)
PY
echo "== v182 $(date +%T)"
python scripts/g_gonderim.py D_kqg_adsb_13a_k115 1.0 b=D_kqg_13a_b5 "v181 ile aynı, düzeltme 1,15 katı (k=1,15; LIRF yankı hedge'i korundu). v181 ile farkı güçlü düzeltmenin 2026 etkisini ölçer." 2>&1 | grep -E "MD5|Dosya|✗"
echo "== v183 $(date +%T)"
python -u scripts/adsb_duzeltici.py uretim havalimani=0 kayip=huber k=1 oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 \
    yanki_beta=0.25 yanki_apt=LIRF kuyruk=1 kapsama=0 ek=_13b 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqg_13b.log
python scripts/g_gonderim.py D_kqg_adsb_13b 1.0 b=D_kqg_13a_b5 "v181 ile aynı, düzeltici kapsama yoğunluğu özellikleri olmadan (ADSB_N, yerel_n, ADSB_BOSLUK, ADSB_BOSLUK5). v181 ile farkı alıcı ağı değişiminin etkisini ölçer." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -3
echo "zincir bitti $(date +%T)"

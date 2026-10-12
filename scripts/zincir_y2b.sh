#!/bin/bash
# 05.10 (YEDEK Y2b): v208 + D1 yalnız LTFM'de k = 1 (eğitime dayalı kural; LFPG k = 0).
# Y1 dosyaları yazıldıktan sonra (numaralar karışmasın diye) gönderim üretir. Kayıt: kutuk/zincir_y2b.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -qE "pist eğitimi başladı|zincir bitti" kutuk/zincir_y1_pist.log 2>/dev/null; do sleep 60; done
python scripts/g_gonderim.py D_kqgpT5_adsb_28a 1.0 b=D_kqgpxg_17a_b5_d1_LTFM1_LFPG0 lobt=1 "v208 + YEDEK Y2b: NM'siz LTFM satırlarına D1 kalıntı düzeltmesi tam ölçekle (k = 1; 721 satır, ort. +60,5 sn), LFPG'de uygulanmadı (2026 düzeltmesi 2025 dağılımının dışında: ort. +7 → +108 sn). Kural eğitim verisinden (2025 ay dışı en iyi k LTFM 1,70–2,85; 1'de kesildi), LB'den değil. + LOBT penceresi. v208 ile farkı = Y2b." 2>&1 | grep -E "Dosya|MD5|model_b" 
echo "zincir bitti $(date +%T)"

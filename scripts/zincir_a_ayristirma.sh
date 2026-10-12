#!/bin/bash
# 09.10 gece (09.10 LB): v305 = 223,5346, v297'ye göre +0,135 → A'da G2 lgb+cb + G2b 2026 Oca / Tem'e
# taşınmadı. v316'nın B'siyle (B5 çıktısı _bdpdcb6c5L, Y1 1σ + pist + LOBT) A tarafı değişkenleri; LB'de A bileşenlerini ayırmak için.
# Önce v316'nın kendisi yeniden kurulur (aynı ayarlar → içerik v316 ile birebir olmalı), sonra değişkenler.
# Kayıt: kutuk/zincir_a_ayristirma.log (+ kutuk/aa_<ad>_{gonderim,kalibre}.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_Y1_ESIK=1.0
B=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5L
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
uret() {  # $1 = kısa ad, $2 = A dosyası, $3 = açıklama
  python scripts/g_gonderim.py $2 1.0 b=$B lobt=1 "A tarafı $2 + v316'nın B'si ($B). $3 Ara dosya." > kutuk/aa_$1_gonderim.log 2>&1 || { echo "HATA $1 gönderim"; exit 1; }
  local ara=$(yol kutuk/aa_$1_gonderim.log)
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 1σ havalimanı + pist + LOBT. v316 ile farkı = yalnız A tarafı: $3" > kutuk/aa_$1_kalibre.log 2>&1 || { echo "HATA $1 kalibre"; exit 1; }
  local son=$(yol kutuk/aa_$1_kalibre.log)
  python - "$son" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v316.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"{sys.argv[1]} − v316: değişen {(np.abs(d) > 1e-3).sum():,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.3f}, maks |fark| {np.abs(d).max():.1f}, min tahmin {b.TAXITIME_SEC_mvt.min():.0f}")
PY
  echo "$1: $ara → $son $(grep -o 'MD5: `[0-9a-f]*`' kutuk/aa_$1_kalibre.log | head -1)"
}
echo "başladı $(date +%T)"
uret v316 D_kqgpPe_adsb_42o_ha_g2lgbcb_g2t "yok (v316'nın yeniden kurulumu; denetim)."
uret g2 D_kqgpPe_adsb_42o_ha_g2 "G2 LightGBM (lgb+cb ve G2b yok; v297'nin A'sı)."
uret g2lgbcb D_kqgpPe_adsb_42o_ha_g2lgbcb "G2 LightGBM + CatBoost, G2b yok (v301'in A'sı)."
uret ha D_kqgpPe_adsb_42o_ha "G2 yok (düzeltici _42o + H1)."
uret duz D_kqgpPe_adsb_42o "G2 ve H1 yok (yalnız düzeltici _42o)."
echo "zincir bitti $(date +%T)"

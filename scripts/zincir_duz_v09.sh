#!/bin/bash
# 11.10: scripts/duz_v_tohum_cv.sh — iki tohum ortalamalı TabM ile
# en iyi v = 0,9 (193,16 → 193,10; Oca −0,09, Tem −0,05; çapraz seçim iki ayda 0,9) → önceden yazılan kural KABUL.
# Üretim: _42p ayarları + nn_w=0.9, 5 tohum tek koşuda (42, 2026, 7, 1, 2), bilesen=1 (rhat_lgb / rhat_nn ayrı) → _42p9 → T1 → _42p9T.
# Adaylar (tek fark v): normal v361'in (A _42p5T → _42p9T), final a64'ün karşılığı.
# LIRF μ zincirinden sonra. Kayıt: kutuk/zincir_duz_v09.log (+ kutuk/dv9_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
until grep -q "zincir bitti\|HATA" kutuk/zincir_lirf_mu_nn.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
duz() {  # $1 = kip, $2 = klasör
  test -f $2/TEST_D_kqgpPeT2_adsb_42p9.parquet || {
    PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 PRC_EPOCH_TAM=1 \
      python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF planli=1 model=lgb+nn nn_w=0.9 nn_epok=8 \
      tohumlar=42,2026,7,1,2 bilesen=1 ek=_42p9 2>&1 | grep -v -i "$F" > kutuk/dv9_duz_$1.log
    [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $1"; }
  PRC_HARMAN=kqgpPeT2 python scripts/hedge_e1_degistir.py D_kqgpPeT2_adsb_42p9 D_kqgpPeT2_adsb_42p9T > kutuk/dv9_t1_$1.log 2>&1 || dur "T1 $1"
  python - "$2" <<'PY'
import sys, numpy as np, pandas as pd
P = sys.argv[1]
a = pd.read_parquet(P + "/TEST_D_kqgpPeT2_adsb_42p5.parquet"); b = pd.read_parquet(P + "/TEST_D_kqgpPeT2_adsb_42p9.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
iz = b.rhat.abs().to_numpy() > 0
r07 = 0.3 * b.rhat_lgb + 0.7 * b.rhat_nn
print(f"  bileşenler: 0,1 · lgb + 0,9 · nn − rhat maks {np.abs(0.1 * b.rhat_lgb + 0.9 * b.rhat_nn - b.rhat).max():.2e} | "
      f"v 0,7 yeniden ↔ _42p5 (izli): korelasyon {np.corrcoef(r07[iz], a.rhat[iz])[0, 1]:.4f}, fark RMS {np.sqrt(np.mean((r07 - a.rhat)[iz] ** 2)):.2f}")
f = b.rhat.to_numpy() - a.rhat.to_numpy()
print(f"  _42p9 − _42p5 rhat (izli {iz.sum():,}): RMS {np.sqrt(np.mean(f[iz] ** 2)):.2f}, ort {f[iz].mean():+.3f}")
PY
}
gonder() {  # $1 = etiket, $2 = B adı, $3 = açıklama, $4 = dosya deseni
  python scripts/g_gonderim.py D_kqgpPeT2_adsb_42p9T 1.0 b=$2 lobt=1 "$3 Ara dosya (Y1'siz)." > kutuk/dv9_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$4" kutuk/dv9_g_$1.log | head -1 | tr '\\' '/'); [ -n "$ara" ] || dur "ara dosya $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. $3" > kutuk/dv9_k_$1.log 2>&1 || dur "Y1 $1"
  SON=$(grep -o "$4" kutuk/dv9_k_$1.log | grep -v "$(basename $ara)" | head -1 | tr '\\' '/')
  echo "$1: ara $ara → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/dv9_k_$1.log | head -1)"
}
# ---------------------------------------------------------------- 1. normal: v361 + düzeltici v 0,9
duz normal data/preds
gonder normal D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5N85 \
  "v361 + düzeltici TabM ağırlığı v 0,7 → 0,9 (A D_kqgpPeT2_adsb_42p9T, 5 tohum). v361 ile tek fark v." \
  "gonderim[\/]strong-moose_v[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim/strong-moose_v361.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"  − v361: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, maks |fark| {np.abs(d).max():.0f}")
PY
# ---------------------------------------------------------------- 2. final: a64 + düzeltici v 0,9
export PRC_FINAL=1
duz final data_final/preds
gonder final D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN85 \
  "FINAL (4 ay): a64 + düzeltici TabM ağırlığı v 0,7 → 0,9 (A D_kqgpPeT2_adsb_42p9T, 5 tohum). a64 ile tek fark v." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
a = pd.read_parquet("gonderim_final/strong-moose_final_a64.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(b.index)
assert len(b) == 670_790 and b.notna().all() and b.min() >= 60
d = (b - a).to_numpy()
print(f"  {sys.argv[1]} − a64: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}")
PY
echo "zincir bitti $(date +%T)"

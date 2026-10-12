#!/bin/bash
# 11.10: düzeltici üretimi 3 → 5 tohum (varyans azaltımı; CV ölçemez, tohum
# ortalaması). _42p ile aynı ayarlar + tohumlar=1,2 → _42pq; scripts/duz_tohum_birlestir.py ile 0,6 · 3 tohum + 0,4 · 2 tohum
# (= tek koşu 5 tohum) → _42p5 → T1 (hedge_e1_degistir.py) → _42p5T. Adaylar (tek fark düzeltici tohum sayısı):
# normal v355'in, final a56'nın karşılığı. B μ varyant zincirinden sonra (bellek). Kayıt: kutuk/zincir_duz_tohum5.log (+ kutuk/dt5_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
until grep -q "zincir bitti" kutuk/b_mu_varyant_cv.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
duz() {  # $1 = kip, $2 = klasör
  test -f $2/TEST_D_kqgpPeT2_adsb_42pq.parquet || {
    PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 PRC_EPOCH_TAM=1 \
      python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF planli=1 model=lgb+nn nn_w=0.7 nn_epok=8 \
      tohumlar=1,2 ek=_42pq 2>&1 | grep -v -i "$F" > kutuk/dt5_duz_$1.log
    [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $1"; }
  PRC_HARMAN=kqgpPeT2 python scripts/duz_tohum_birlestir.py D_kqgpPeT2_adsb_42p D_kqgpPeT2_adsb_42pq D_kqgpPeT2_adsb_42p5 \
    > kutuk/dt5_birlestir_$1.log 2>&1 || dur "birleştirme $1"
  PRC_HARMAN=kqgpPeT2 python scripts/hedge_e1_degistir.py D_kqgpPeT2_adsb_42p5 D_kqgpPeT2_adsb_42p5T > kutuk/dt5_t1_$1.log 2>&1 || dur "T1 $1"
  cat kutuk/dt5_birlestir_$1.log | grep -v yazıldı
}
gonder() {  # $1 = etiket, $2 = B adı, $3 = açıklama, $4 = dosya deseni
  python scripts/g_gonderim.py D_kqgpPeT2_adsb_42p5T 1.0 b=$2 lobt=1 "$3 Ara dosya (Y1'siz)." > kutuk/dt5_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$4" kutuk/dt5_g_$1.log | head -1 | tr '\' '/'); [ -n "$ara" ] || dur "ara dosya $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. $3" > kutuk/dt5_k_$1.log 2>&1 || dur "Y1 $1"
  SON=$(grep -o "$4" kutuk/dt5_k_$1.log | grep -v "$(basename $ara)" | head -1 | tr '\' '/')
  echo "$1: ara $ara → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/dt5_k_$1.log | head -1)"
}
# ---------------------------------------------------------------- 1. normal: v355 + düzeltici 5 tohum
duz normal data/preds
gonder normal D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5N8 \
  "v355 + düzeltici 5 tohum (3 + 2; A D_kqgpPeT2_adsb_42p5T). v355 ile tek fark düzelticinin üretim tohum sayısı." \
  "gonderim[\/]strong-moose_v[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim/strong-moose_v355.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"  − v355: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, maks |fark| {np.abs(d).max():.0f}")
PY
# ---------------------------------------------------------------- 2. final: a56 + düzeltici 5 tohum
export PRC_FINAL=1
duz final data_final/preds
gonder final D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN8 \
  "FINAL (4 ay): a56 + düzeltici 5 tohum (3 + 2; A D_kqgpPeT2_adsb_42p5T). a56 ile tek fark düzelticinin üretim tohum sayısı." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim_final/strong-moose_final_a56.parquet")
assert len(b) == 670_790 and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_t"))
x["f"] = x.TAXITIME_SEC_mvt_t - x.TAXITIME_SEC_mvt
print(f"  {sys.argv[1]} − a56: değişen {(x.f.abs() > 1e-3).sum():,}, RMS {np.sqrt((x.f ** 2).mean()):.2f}, ort {x.f.mean():+.3f}")
PY
echo "zincir bitti $(date +%T)"

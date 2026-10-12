#!/bin/bash
# 10.10: TabM 2 tohum × 12 epok (12 ay OOF düzeltici sonrası −1,030 → −1,208,
# w = 0,40) → harman kqgpPeT2 → düzeltici _42o → adaylar.
#   0) düzeltici CV (Oca / Tem, kqgpPeT2; karşılık dcv_duz_parkb_tabm.log 194,97) — işlemcide, test TabM ile paralel
#   1) test TabM (final özellikleri, 2 tohum × 12 epok): normal → final  [ekran kartı]
#   2) normal kip: A = D_kqgpPeT2_adsb_42o + 8 havalimanlı B (v343'ün karşılığı; v343 ile tek fark TabM'in güçlenmesi)
#   3) final kip: 6 / 8 havalimanlı B (a42 / a44'ün karşılıkları)
# Kayıt: kutuk/zincir_tabm2_aday.log (+ kutuk/t2_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
echo "başladı $(date +%T)"
for s in 1 2 3 4 5 6; do test -f data_final/preds/NN_tabm_kqgp2_split_$s.parquet || dur "data_final'de TabM2 OOF yok"; done
# ---------------------------------------------------------------- 0. düzeltici CV (paralel)
( PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 \
    python -u scripts/adsb_duzeltici.py cv $D parkb=1 2>&1 | grep -v -i "$F" > kutuk/dcv_duz_parkb_tabm2.log
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_parkb_tabm2.parquet
  echo "CV: $(grep 'k=1.0 ' kutuk/dcv_duz_parkb_tabm2.log)" ) &
CVPID=$!
# ---------------------------------------------------------------- 1. test TabM
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_final.json
T="model=tabm tabm_k=8 amp=1 epok=12 tohum=42,7"
python -u scripts/nn_bilesen.py tabm_kqgpy_e12 tam $T > kutuk/t2_test_normal.log 2>&1 || dur "TabM test normal"
echo "TabM test normal tamam $(date +%T)"
PRC_FINAL=1 python -u scripts/nn_bilesen.py tabm_kqgpy_e12 tam $T > kutuk/t2_test_final.log 2>&1 &
TFPID=$!
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ PRC_EKOZ_DOSYA
duz() {  # $1 = kip
  PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 PRC_EPOCH_TAM=1 \
    python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_42o 2>&1 | grep -v -i "$F" > kutuk/t2_duz_$1.log
  [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $1"
}
gonder() {  # $1 = etiket, $2 = B adı, $3 = açıklama, $4 = dosya deseni
  python scripts/g_gonderim.py D_kqgpPeT2_adsb_42o 1.0 b=$2 lobt=1 "$3 Ara dosya (Y1'siz)." > kutuk/t2_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$4" kutuk/t2_g_$1.log | head -1 | tr '\\' '/'); [ -n "$ara" ] || dur "ara dosya $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. $3" > kutuk/t2_k_$1.log 2>&1 || dur "Y1 $1"
  SON=$(grep -o "$4" kutuk/t2_k_$1.log | head -1 | tr '\\' '/')
  echo "$1: ara $ara → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/t2_k_$1.log | head -1)"
}
# ---------------------------------------------------------------- 2. normal kip
python scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPeT2.json D_kqgpPeT2 > kutuk/t2_harman_normal.log 2>&1 || dur "harman normal"
duz normal
echo "normal düzeltici tamam $(date +%T)"
gonder normal D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5L8 \
  "A = D_kqgpPeT2_adsb_42o (TabM 2 tohum × 12 epok, w 0,40 + düzeltici _42o; H1 / G2 yok) + 8 havalimanlı B. v343 ile tek fark TabM'in güçlenmesi." \
  "gonderim[\/]strong-moose_v[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1])
for k in ("v343", "v341"):
    a = pd.read_parquet(f"gonderim/strong-moose_{k}.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  − {k}: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, min {b.TAXITIME_SEC_mvt.min():.0f}")
PY
# ---------------------------------------------------------------- 3. final kip
wait $TFPID || dur "TabM test final"
echo "TabM test final tamam $(date +%T)"
export PRC_FINAL=1
python scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPeT2.json D_kqgpPeT2 > kutuk/t2_harman_final.log 2>&1 || dur "harman final"
duz final
echo "final düzeltici tamam $(date +%T)"
gonder final6 D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bF \
  "FINAL (4 ay): A = D_kqgpPeT2_adsb_42o (TabM 2 tohum × 12 epok + düzeltici _42o; H1 / G2 yok), 6 havalimanlı B. a42 ile tek fark TabM'in güçlenmesi." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
gonder final8 D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bF8 \
  "FINAL (4 ay): A = D_kqgpPeT2_adsb_42o (TabM 2 tohum × 12 epok + düzeltici _42o; H1 / G2 yok), 8 havalimanlı B. a44 ile tek fark TabM'in güçlenmesi." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim_final/strong-moose_final_a44.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "MVT_MONTH"])
assert len(b) == 670_790 and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_t")).merge(r, on="MVT_ID_mvt")
x["f"] = x.TAXITIME_SEC_mvt_t - x.TAXITIME_SEC_mvt
print(f"  {sys.argv[1]} − a44: RMS {np.sqrt((x.f ** 2).mean()):.2f}, ort {x.f.mean():+.2f}")
print(x.groupby("MVT_MONTH").f.agg(["mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(2).to_string())
PY
wait $CVPID
echo "zincir bitti $(date +%T)"

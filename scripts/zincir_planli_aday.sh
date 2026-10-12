#!/bin/bash
# 10.10: düzelticiye planlı saatler (planli=1) yeni düzeltici
# üzerinde CV Oca / Tem 193,311 → 193,121 (−0,19; Oca −0,25, Tem −0,16 → KABUL; scripts/duz_kiyas.py, kutuk/duz_planli_cv.log).
# Üretim: _42n ile aynı + planli=1 → ek _42p. Adaylar (tek fark planli): normal v349 + planli; final a52 + planli.
# B5 k CV'si (b5_k_cv.sh) bitmeden başlamaz (bellek). Kayıt: kutuk/zincir_planli_aday.log (+ kutuk/dp_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
until grep -q "^bitti\|HATA" kutuk/b5_k_cv.log 2>/dev/null; do sleep 30; done
echo "başladı $(date +%T)"
duz() {  # $1 = kip
  test -f $2/TEST_D_kqgpPeT2_adsb_42p.parquet && { echo "düzeltici $1 mevcut"; return 0; }
  PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 PRC_EPOCH_TAM=1 \
    python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF planli=1 model=lgb+nn nn_w=0.7 nn_epok=8 ek=_42p \
    2>&1 | grep -v -i "$F" > kutuk/dp_duz_$1.log
  [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $1"
  python - "$2" <<'PY'
import sys, numpy as np, pandas as pd
P = sys.argv[1]
a = pd.read_parquet(P + "/TEST_D_kqgpPeT2_adsb_42n.parquet"); b = pd.read_parquet(P + "/TEST_D_kqgpPeT2_adsb_42p.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
f = b.taksi.to_numpy(float) - a.taksi.to_numpy(float)
iz = a.rhat.abs().to_numpy() > 1e-9
print(f"  _42p − _42n: izli {iz.sum():,} satır RMS {np.sqrt(np.mean(f[iz] ** 2)):.1f}, ort {f[iz].mean():+.2f}; izsiz maks |fark| {np.abs(f[~iz]).max():.3f}")
print(f"  rhat korelasyonu {np.corrcoef(a.rhat[iz], b.rhat[iz])[0, 1]:.4f}")
PY
}
gonder() {  # $1 = etiket, $2 = B adı, $3 = açıklama, $4 = dosya deseni
  python scripts/g_gonderim.py D_kqgpPeT2_adsb_42p 1.0 b=$2 lobt=1 "$3 Ara dosya (Y1'siz)." > kutuk/dp_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$4" kutuk/dp_g_$1.log | head -1 | tr '\\' '/'); [ -n "$ara" ] || dur "ara dosya $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. $3" > kutuk/dp_k_$1.log 2>&1 || dur "Y1 $1"
  SON=$(grep -o "$4" kutuk/dp_k_$1.log | head -1 | tr '\\' '/')
  echo "$1: ara $ara → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/dp_k_$1.log | head -1)"
}
# ---------------------------------------------------------------- 1. normal kip: v349 + planli
duz normal data/preds
echo "normal düzeltici tamam $(date +%T)"
gonder normal D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5N8 \
  "A = D_kqgpPeT2_adsb_42p (TabM harman + düzeltici LightGBM + TabM, v 0,7, planlı saatler) + 8 havalimanlı B, B μ'ye TabM. v349 ile tek fark planli." \
  "gonderim[\/]strong-moose_v[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1])
for k in ("v349", "v351"):
    a = pd.read_parquet(f"gonderim/strong-moose_{k}.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  − {k}: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, min {b.TAXITIME_SEC_mvt.min():.0f}")
PY
# ---------------------------------------------------------------- 2. final kip: a52 + planli
export PRC_FINAL=1
duz final data_final/preds
echo "final düzeltici tamam $(date +%T)"
gonder final D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN8 \
  "FINAL (4 ay): A = D_kqgpPeT2_adsb_42p (TabM harman + düzeltici LightGBM + TabM, v 0,7, planlı saatler) + 8 havalimanlı B, B μ'ye TabM. a52 ile tek fark planli." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim_final/strong-moose_final_a52.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "MVT_MONTH"])
assert len(b) == 670_790 and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_t")).merge(r, on="MVT_ID_mvt")
x["f"] = x.TAXITIME_SEC_mvt_t - x.TAXITIME_SEC_mvt
print(f"  {sys.argv[1]} − a52: RMS {np.sqrt((x.f ** 2).mean()):.2f}, ort {x.f.mean():+.2f}")
print(x.groupby("MVT_MONTH").f.agg(["mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(2).to_string())
PY
echo "zincir bitti $(date +%T)"

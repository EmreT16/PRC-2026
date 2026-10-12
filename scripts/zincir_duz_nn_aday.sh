#!/bin/bash
# 10.10: CV (Oca / Tem, taban kqgpPeT2) LightGBM 194,75 → LightGBM + TabM
# (v = 0,7) 193,31 (−1,44; çapraz Tem −1,36, Oca −1,47 → KABUL; kutuk/duz_nn_degerlendir.log). Üretimde 3 tohum × (LightGBM +
# TabM 8 epok), rhat = 0,3 · LightGBM + 0,7 · TabM; ayarlar _42o ile birebir (k = 1, LIRF yankı β 0,25) → ek _42n.
# Adaylar (tek fark düzeltici): normal v347 + TabM düzeltici; final a50 + TabM düzeltici. PYTHONHASHSEED=0 (koşudan koşuya gürültü).
# Kayıt: kutuk/zincir_duz_nn_aday.log (+ kutuk/dn_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
echo "başladı $(date +%T)"
duz() {  # $1 = kip
  test -f $2/TEST_D_kqgpPeT2_adsb_42n.parquet && { echo "düzeltici $1 mevcut"; return 0; }
  PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2 PRC_EPOCH_TAM=1 \
    python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF model=lgb+nn nn_w=0.7 nn_epok=8 ek=_42n \
    2>&1 | grep -v -i "$F" > kutuk/dn_duz_$1.log
  [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $1"
  python - "$2" <<'PY'
import sys, numpy as np, pandas as pd
P = sys.argv[1]
a = pd.read_parquet(P + "/TEST_D_kqgpPeT2_adsb_42o.parquet"); b = pd.read_parquet(P + "/TEST_D_kqgpPeT2_adsb_42n.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
f = b.taksi.to_numpy(float) - a.taksi.to_numpy(float)
iz = a.rhat.abs().to_numpy() > 1e-9
print(f"  _42n − _42o: izli {iz.sum():,} satır RMS {np.sqrt(np.mean(f[iz] ** 2)):.1f}, ort {f[iz].mean():+.2f}; izsiz maks |fark| {np.abs(f[~iz]).max():.3f}")
print(f"  rhat korelasyonu {np.corrcoef(a.rhat[iz], b.rhat[iz])[0, 1]:.4f}")
PY
}
gonder() {  # $1 = etiket, $2 = B adı, $3 = açıklama, $4 = dosya deseni
  python scripts/g_gonderim.py D_kqgpPeT2_adsb_42n 1.0 b=$2 lobt=1 "$3 Ara dosya (Y1'siz)." > kutuk/dn_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$4" kutuk/dn_g_$1.log | head -1 | tr '\\' '/'); [ -n "$ara" ] || dur "ara dosya $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. $3" > kutuk/dn_k_$1.log 2>&1 || dur "Y1 $1"
  SON=$(grep -o "$4" kutuk/dn_k_$1.log | head -1 | tr '\\' '/')
  echo "$1: ara $ara → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/dn_k_$1.log | head -1)"
}
# ---------------------------------------------------------------- 1. normal kip: v347 + TabM düzeltici
duz normal data/preds
echo "normal düzeltici tamam $(date +%T)"
gonder normal D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5N8 \
  "A = D_kqgpPeT2_adsb_42n (TabM harman + düzeltici LightGBM + TabM, v 0,7) + 8 havalimanlı B, B μ'ye TabM. v347 ile tek fark düzeltici." \
  "gonderim[\/]strong-moose_v[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1])
for k in ("v347", "v345"):
    a = pd.read_parquet(f"gonderim/strong-moose_{k}.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  − {k}: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, min {b.TAXITIME_SEC_mvt.min():.0f}")
PY
# ---------------------------------------------------------------- 2. final kip: a50 + TabM düzeltici
export PRC_FINAL=1
duz final data_final/preds
echo "final düzeltici tamam $(date +%T)"
gonder final D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN8 \
  "FINAL (4 ay): A = D_kqgpPeT2_adsb_42n (TabM harman + düzeltici LightGBM + TabM, v 0,7) + 8 havalimanlı B, B μ'ye TabM. a50 ile tek fark düzeltici." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim_final/strong-moose_final_a50.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "MVT_MONTH"])
assert len(b) == 670_790 and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_t")).merge(r, on="MVT_ID_mvt")
x["f"] = x.TAXITIME_SEC_mvt_t - x.TAXITIME_SEC_mvt
print(f"  {sys.argv[1]} − a50: RMS {np.sqrt((x.f ** 2).mean()):.2f}, ort {x.f.mean():+.2f}")
print(x.groupby("MVT_MONTH").f.agg(["mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(2).to_string())
PY
echo "zincir bitti $(date +%T)"

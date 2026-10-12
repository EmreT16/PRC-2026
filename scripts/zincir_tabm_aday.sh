#!/bin/bash
# 09.10 gece: TabM'li harman (kqgpPeT = kqgpPe × 0,6 + TabM × 0,4) →
# Model A düzelticisi (_42o ayarlarıyla yeniden eğitim) → gönderim. Düzeltici CV Oca / Tem 196,14 → 194,97.
#   1) normal kip: LB adayı = v330'un TabM'li karşılığı (A yalnız düzeltici, H1 / G2 yok; B v316'nınki) → v330 ile tek fark TabM
#   2) final kip: 6 ve 8 havalimanlı B ile final adayları (a30 / a40'ın TabM'li karşılıkları)
# Test TabM tahminleri scripts/tabm_test_kuyruk.sh'den (kutuk/tabm_test.log). Kayıt: kutuk/zincir_tabm_aday.log (+ kutuk/ta_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_Y1_ESIK=1.0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
duz() {  # $1 = kip etiketi; PRC_FINAL dışarıdan
  PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT PRC_EPOCH_TAM=1 \
    python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_42o 2>&1 | grep -v -i "$F" > kutuk/ta_duz_$1.log
  [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $1"
}
gonder() {  # $1 = etiket, $2 = B adı, $3 = açıklama, $4 = gonderim klasörü deseni
  python scripts/g_gonderim.py D_kqgpPeT_adsb_42o 1.0 b=$2 lobt=1 "$3 Ara dosya (Y1'siz)." > kutuk/ta_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$4" kutuk/ta_g_$1.log | head -1 | tr '\\' '/'); [ -n "$ara" ] || dur "ara dosya $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. $3" > kutuk/ta_k_$1.log 2>&1 || dur "Y1 $1"
  local son=$(grep -o "$4" kutuk/ta_k_$1.log | head -1 | tr '\\' '/')
  echo "$1: ara $ara → $son $(grep -o 'MD5: `[0-9a-f]*`' kutuk/ta_k_$1.log | head -1)"
  SON=$son
}
echo "başladı $(date +%T)"
# ---------------------------------------------------------------- 1. normal kip (Oca / Tem LB); "final" argümanıyla atlanır
if [ "$1" != "final" ]; then
until grep -qE "final kip başladı|HATA" kutuk/tabm_test.log 2>/dev/null; do sleep 60; done
grep -q "HATA normal" kutuk/tabm_test.log && dur "TabM test (normal)"
python scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPeT.json D_kqgpPeT > kutuk/ta_harman_normal.log 2>&1 || dur "test harmanı (normal)"
echo "normal harman: $(grep -o 'ort. taksi [0-9.]*' kutuk/ta_harman_normal.log)"
duz normal
echo "normal düzeltici tamam $(date +%T)"
gonder normal D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5L \
  "A = D_kqgpPeT_adsb_42o (TabM'li harman kqgpPeT + düzeltici _42o; H1 / G2 yok) + v316'nın B'si. v330 ile tek fark TabM." \
  "gonderim[\/]strong-moose_v[0-9]*\.parquet"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1])
for k in ("v330", "v316"):
    a = pd.read_parquet(f"gonderim/strong-moose_{k}.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  − {k}: değişen {(np.abs(d) > 1e-3).sum():,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, min tahmin {b.TAXITIME_SEC_mvt.min():.0f}")
PY
fi
# ---------------------------------------------------------------- 2. final kip (4 ay). OOF tarafı data_final/preds'ten okunur:
# NN_tabm_kqgp_split_* final_hazirlik.py'nin kuralıyla (data/preds'teki TEST_ dışı dosyalar) oraya kopyalanmış olmalı
for s in 1 2 3 4 5 6; do test -f data_final/preds/NN_tabm_kqgp_split_$s.parquet || dur "data_final'de TabM OOF yok (final_hazirlik.py)"; done
until grep -qE "^bitti|HATA final" kutuk/tabm_test.log 2>/dev/null; do sleep 60; done
grep -q "HATA final" kutuk/tabm_test.log && dur "TabM test (final)"
export PRC_FINAL=1
python scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPeT.json D_kqgpPeT > kutuk/ta_harman_final.log 2>&1 || dur "test harmanı (final)"
echo "final harman: $(grep -o 'ort. taksi [0-9.]*' kutuk/ta_harman_final.log)"
duz final
echo "final düzeltici tamam $(date +%T)"
gonder final6 D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bF \
  "FINAL (4 ay): A = D_kqgpPeT_adsb_42o (TabM'li harman + düzeltici _42o; H1 / G2 yok), 6 havalimanlı B. a30 ile tek fark TabM." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
F6=$SON
gonder final8 D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bF8 \
  "FINAL (4 ay): A = D_kqgpPeT_adsb_42o (TabM'li harman + düzeltici _42o; H1 / G2 yok), 8 havalimanlı B. a40 ile tek fark TabM." \
  "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet"
python - "$F6" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim_final/strong-moose_final_a30.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
assert len(b) == 670_790 and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_t")).merge(r, on="MVT_ID_mvt")
x["f"] = x.TAXITIME_SEC_mvt_t - x.TAXITIME_SEC_mvt
print(f"  {sys.argv[1]} − a30: RMS {np.sqrt((x.f ** 2).mean()):.2f}, ort {x.f.mean():+.2f}")
print(x.groupby(["MVT_MONTH"]).f.agg(["mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(2).to_string())
PY
echo "zincir bitti $(date +%T)"

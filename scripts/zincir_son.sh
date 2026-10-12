#!/bin/bash
# 11.10 öğle: LB tarafının son iki işi, düzeltici v 0,9 zincirinden sonra.
# (1) Düzeltici v, 3 tohum: TabM-only CV tohum 2026 (planlısız, v ızgarası düzeni) → rn = ort(42, 7, 2026) →
#     duz_nn_degerlendir.py ref_v=0.9 (kural: v* ≠ 0,9, ağırlıklı ≥ 0,05 ve iki ay iyi). KABUL → _42p9'un bileşenlerinden
#     yeniden eğitimsiz (scripts/duz_v_degistir.py) aday: normal v365'in, final a(v 0,9)'un karşılığı.
# (2) Taban TabM 5 tohum: test TabM'e 3 tohum daha (2026, 1, 2; normal + final) → 0,4 · e12 + 0,6 · e12t3 → harman
#     D_kqgpPeT2t5 (ağırlıklar / OOF aynı) → düzeltici (5 tohum, v 0,9, bilesen=1; seçilen v'ye duz_v_degistir) → T1 → adaylar
#     (tek fark taban TabM tohum sayısı): normal / final, (1)'in sonucu dahil en iyi A'ya karşı.
# Kayıt: kutuk/zincir_son.log (+ kutuk/zs_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
BN=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5N85
BF=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN85
until grep -q "zincir bitti\|HATA" kutuk/zincir_duz_v09.log 2>/dev/null; do sleep 60; done
grep -q "HATA" kutuk/zincir_duz_v09.log && dur "v 0,9 zinciri hatalı"
V9N=$(grep "^normal:" kutuk/zincir_duz_v09.log | grep -o "strong-moose_v[0-9]*\.parquet" | tail -1)
V9F=$(grep "^final:" kutuk/zincir_duz_v09.log | grep -o "strong-moose_final_a[0-9]*\.parquet" | tail -1)
echo "başladı $(date +%T) | v 0,9 adayları: $V9N / $V9F"
gonder() {  # $1 = etiket, $2 = A adı, $3 = B adı, $4 = açıklama, $5 = dosya deseni, $6 = karşılık dosyası
  python scripts/g_gonderim.py $2 1.0 b=$3 lobt=1 "$4 Ara dosya (Y1'siz)." > kutuk/zs_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$5" kutuk/zs_g_$1.log | head -1 | tr '\\' '/'); [ -n "$ara" ] || dur "ara dosya $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. $4" > kutuk/zs_k_$1.log 2>&1 || dur "Y1 $1"
  SON=$(grep -o "$5" kutuk/zs_k_$1.log | grep -v "$(basename $ara)" | head -1 | tr '\\' '/')
  echo "$1: ara $ara → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/zs_k_$1.log | head -1)"
  python - "$SON" "$6" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
a = pd.read_parquet(sys.argv[2]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(b.index)
assert len(b) in (344_841, 670_790) and b.notna().all() and b.min() >= 60 and a.notna().all()
d = (b - a).to_numpy()
print(f"  − {sys.argv[2].split('/')[-1]}: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, maks |fark| {np.abs(d).max():.0f}")
PY
}
vdegis() {  # $1 = harman öneki (D_kqgpPeT2 | D_kqgpPeT2t5), $2 = PRC_HARMAN → A adı (T1'li) yazar: AV
  if [ "$V" = "0.9" ]; then AV=${1}_adsb_42p9T; return 0; fi
  PRC_HARMAN=$2 python scripts/duz_v_degistir.py ${1}_adsb_42p9 ${1}_adsb_42pv v=$V v0=0.9 > kutuk/zs_v_$2_${PRC_FINAL:-0}.log 2>&1 || dur "v değişimi $1"
  PRC_HARMAN=$2 python scripts/hedge_e1_degistir.py ${1}_adsb_42pv ${1}_adsb_42pvT > kutuk/zs_vt1_$2_${PRC_FINAL:-0}.log 2>&1 || dur "T1 $1"
  AV=${1}_adsb_42pvT
}
# ================================================================ (1) düzeltici v, 3 tohum
export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1
test -f data/preds/ADSB_duz_cv_parkb_tabm2_nn_s2026.parquet || {
  PRC_HARMAN=kqgpPeT2 python -u scripts/adsb_duzeltici.py cv $D parkb=1 model=nn nn_epok=8 cv_tohum=2026 2>&1 | grep -v -i "$F" > kutuk/dcv_duz_parkb_tabm2_nn_s2026.log
  [ ${PIPESTATUS[0]} -eq 0 ] || dur "CV tohum 2026"
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_parkb_tabm2_nn_s2026.parquet; }
python - <<'PY' || dur "3 tohum ortalaması"
import numpy as np, pandas as pd
P = "data/preds/ADSB_duz_cv_parkb_tabm2_nn"
x = pd.read_parquet(P + ".parquet")[["MVT_ID_mvt", "rhat"]]
for s in ("_s7", "_s2026"):
    x = x.merge(pd.read_parquet(P + s + ".parquet")[["MVT_ID_mvt", "rhat"]].rename(columns={"rhat": s}), on="MVT_ID_mvt", validate="one_to_one")
x["rhat"] = (x.rhat + x._s7 + x._s2026) / 3
x[["MVT_ID_mvt", "rhat"]].to_parquet(P + "_ort3.parquet", index=False)
PY
PRC_HARMAN=kqgpPeT2 python scripts/duz_nn_degerlendir.py nn=ADSB_duz_cv_parkb_tabm2_nn_ort3 ref_v=0.9 2>&1 | grep -v -i warn > kutuk/zs_v3.log
cut -c1-50 kutuk/zs_v3.log; tail -n 2 kutuk/zs_v3.log
V=$(grep -o "V_SECIM=[0-9.]*" kutuk/zs_v3.log | cut -d= -f2); [ -n "$V" ] || dur "v seçimi"
echo "seçilen v = $V $(date +%T)"
unset PRC_ADSB_OZ PRC_ADSB_GEVSEK PRC_DUZ_EK PRC_EPOCH_TAM
if [ "$V" != "0.9" ]; then
  vdegis D_kqgpPeT2 kqgpPeT2
  gonder vn $AV $BN "v365 + düzeltici v 0,9 → $V (3 tohum CV; yeniden eğitimsiz, bileşenlerden). v365 ile tek fark v." "gonderim[\/]strong-moose_v[0-9]*\.parquet" gonderim/$V9N
  REFN=$SON
  export PRC_FINAL=1; vdegis D_kqgpPeT2 kqgpPeT2
  gonder vf $AV $BF "FINAL (4 ay): $V9F + düzeltici v $V. Tek fark v." "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" gonderim_final/$V9F
  REFF=$SON; unset PRC_FINAL
else
  REFN=gonderim/$V9N; REFF=gonderim_final/$V9F
fi
# ================================================================ (2) taban TabM 5 tohum
TT="PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_final.json"
for KIP in normal final; do
  if [ $KIP = final ]; then export PRC_FINAL=1; P=data_final/preds; else unset PRC_FINAL; P=data/preds; fi
  test -f $P/TEST_nn_tabm_kqgpy_e12t3.parquet || env $TT python -u scripts/nn_bilesen.py tabm_kqgpy_e12t3 tam model=tabm tabm_k=8 amp=1 epok=12 tohum=2026,1,2 \
    > kutuk/zs_tabm_$KIP.log 2>&1 || dur "test TabM $KIP"
  python - "$P" <<'PY' || dur "TabM ortalaması"
import sys, numpy as np, pandas as pd
P = sys.argv[1]
a = pd.read_parquet(f"{P}/TEST_nn_tabm_kqgpy_e12.parquet"); b = pd.read_parquet(f"{P}/TEST_nn_tabm_kqgpy_e12t3.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
print(f"  {P}: 2 ↔ 3 tohum korelasyonu {np.corrcoef(a.nn, b.nn)[0, 1]:.5f}, fark RMS {np.sqrt(np.mean((a.nn - b.nn) ** 2)):.2f}")
a["nn"] = 0.4 * a.nn.to_numpy() + 0.6 * b.nn.to_numpy()
a.to_parquet(f"{P}/TEST_nn_tabm_kqgpy_e12t5.parquet", index=False)
PY
  PRC_TABM_T2_TEST=TEST_nn_tabm_kqgpy_e12t5 python scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPeT2.json D_kqgpPeT2t5 > kutuk/zs_harman_$KIP.log 2>&1 || dur "harman $KIP"
  test -f $P/TEST_D_kqgpPeT2t5_adsb_42p9.parquet || {
    PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT2t5 PRC_EPOCH_TAM=1 \
      python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF planli=1 model=lgb+nn nn_w=0.9 nn_epok=8 \
      tohumlar=42,2026,7,1,2 bilesen=1 ek=_42p9 2>&1 | grep -v -i "$F" > kutuk/zs_duz_$KIP.log
    [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $KIP"; }
  PRC_HARMAN=kqgpPeT2t5 python scripts/hedge_e1_degistir.py D_kqgpPeT2t5_adsb_42p9 D_kqgpPeT2t5_adsb_42p9T > kutuk/zs_t1_$KIP.log 2>&1 || dur "T1 $KIP"
  vdegis D_kqgpPeT2t5 kqgpPeT2t5
  echo "$KIP A hazır: $AV $(date +%T)"
  if [ $KIP = normal ]; then
    gonder tn $AV $BN "$(basename $REFN) + taban TabM 5 tohum (harman D_kqgpPeT2t5; düzeltici v $V). Tek fark taban TabM tohum sayısı." "gonderim[\/]strong-moose_v[0-9]*\.parquet" $REFN
  else
    gonder tf $AV $BF "FINAL (4 ay): $(basename $REFF) + taban TabM 5 tohum (düzeltici v $V). Tek fark taban TabM tohum sayısı." "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" $REFF
  fi
done
echo "zincir bitti $(date +%T)"

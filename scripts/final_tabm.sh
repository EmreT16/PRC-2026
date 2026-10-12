#!/bin/bash
# 10.10: final hattının TabM aşaması — final_tum.sh'nin son adımı (final_son.sh +
# final_b8.sh'den sonra). Son final yapısı:
#   A = harman kqgpPeT2 (5 ağaç bileşeni × 0,6 + TabM 12 epok × 0,4) → ADS-B düzeltici (k = 1, LIRF yankı β 0,25, planlı saatler;
#       LightGBM + TabM) → T1 (LIRF yankı hedge'inin p_kilit'i final E1'den); ikinci aşama (H1 / G2 / G2b) yok.
#   B = 8 havalimanlı yeni B (EDDF v126 tabanında), μ = 0,4 · melez + 0,6 · TabM, LIRF düzeni ayrı → B5 (k = 1,25)
#       → gönderim + Y1 (havalimanı + pist 1σ) + LOBT.
# 11.10 eklemeleri (her biri bayrakla kapatılabilir; varsayılan 1 = kabul edilen yapı):
#   FT_TABM5=1   taban test TabM 5 tohum (42, 7 + 2026, 1, 2) → harman D_kqgpPeT2t5 (ağırlıklar / OOF aynı)
#   FT_DUZ_V9=1  düzeltici TabM ağırlığı v 0,9, 5 tohum tek koşu, bileşenler ayrı (_42p9); 0 → v 0,7, 3 tohum (_42p)
#   FT_BMU5=1    B μ TabM 5 tohum (0,4 · t2 + 0,6 · t3 → t5; B5 adı _bN85)
#   FT_LIRF_MU=1 LIRF μ'ye kısmi TabM (p0 > 0,95, w 0,6; scripts/b_lirf_mu_nn_uygula.py) + 24 sa karışımı yeniden (B5 adı …L)
#   FT_YETIM=1   yetim B penceresi (LIRF / LTFM; varış kaydından LOBT ± 3.606 dışlama; scripts/b_yetim_pencere.py)
#   FT_GUNLUK=1  günlük kalkış öncesi gecikme kısıtı (Y1 + LOBT sonrası; yalnız A, LIRF hariç, ±300 sn; scripts/g_gunluk.py)
#   Eski bayraklar: FT_DUZ_NN / FT_PLANLI / FT_T1 (varsayılan 1), FT_B5 (TabM'li B5, varsayılan yok).
# Önkoşullar (eğitim tarafı, normal hat): data/preds/NN_tabm_kqgp2_split_1…6 (scripts/tabm_guclendir.sh; final_hazirlik.py
# data_final/preds'e kopyalar), kutuk/g1_agirliklar_kqgpPeT2.json; final_son.sh'nin B / LIRF dosyaları (TEST_B_tum_dpdcb6c5,
# TEST_B_tum6M_adsb, TEST_B_tum6M_adsb_lirfte_mu2(_tk), TEST_B_lirf_te_mu2, TEST_B_tum_dpdcb3ham_hepsi).
# Kayıt: kutuk/final_tabm.log (+ kutuk/ftm_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_FINAL=1 PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
DUZ_NN=${FT_DUZ_NN:-1}
PLANLI=${FT_PLANLI:-1}  # 10.10: düzelticiye planlı saatler (CV −0,19; v353 / a54)
TABM5=${FT_TABM5:-1}
V9=${FT_DUZ_V9:-1}
BMU5=${FT_BMU5:-1}
LIRF=${FT_LIRF_MU:-1}
B5=${FT_B5:-}
F='warn\|X\["tip"\]'
P=data_final/preds
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
yol() { grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" "$1" | head -1 | tr '\\' '/'; }
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
TT="PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_final.json"
echo "başladı $(date +%T) (düzeltici TabM: $DUZ_NN, planli: $PLANLI, taban TabM 5: $TABM5, v 0,9: $V9, B μ 5: $BMU5, LIRF μ: $LIRF, B5: ${B5:-lgb})"
for s in 1 2 3 4 5 6; do test -f $P/NN_tabm_kqgp2_split_$s.parquet || dur "TabM OOF yok (split_$s)"; done
# ---------------------------------------------------------------- 1. test TabM (final özellikleri; ekran kartı)
test -f $P/TEST_nn_tabm_kqgpy_e12.parquet || env $TT python -u scripts/nn_bilesen.py tabm_kqgpy_e12 tam model=tabm tabm_k=8 amp=1 epok=12 tohum=42,7 \
  > kutuk/ftm_tabm.log 2>&1 || dur "test TabM"
HAR=kqgpPeT2
if [ "$TABM5" = "1" ]; then  # 11.10: 3 tohum daha, 0,4 · e12 + 0,6 · e12t3 (= 5 tohum ortalaması)
  test -f $P/TEST_nn_tabm_kqgpy_e12t3.parquet || env $TT python -u scripts/nn_bilesen.py tabm_kqgpy_e12t3 tam model=tabm tabm_k=8 amp=1 epok=12 \
    tohum=2026,1,2 > kutuk/ftm_tabm_t3.log 2>&1 || dur "test TabM t3"
  python - <<'PY' || dur "TabM ortalaması"
import pandas as pd
P = "data_final/preds/"
a = pd.read_parquet(P + "TEST_nn_tabm_kqgpy_e12.parquet"); b = pd.read_parquet(P + "TEST_nn_tabm_kqgpy_e12t3.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
a["nn"] = 0.4 * a.nn.to_numpy() + 0.6 * b.nn.to_numpy()
a.to_parquet(P + "TEST_nn_tabm_kqgpy_e12t5.parquet", index=False)
PY
  HAR=kqgpPeT2t5
fi
echo "test TabM tamam $(date +%T) (harman $HAR)"
# ---------------------------------------------------------------- 2. harman
HAD=$(python -c "import os; os.environ['PRC_HARMAN']='$HAR'; import sys; sys.path.insert(0, '.'); from src import adsb; print(adsb.TEST_AD)")
PRC_TABM_T2_TEST=$( [ "$TABM5" = "1" ] && echo TEST_nn_tabm_kqgpy_e12t5 || echo TEST_nn_tabm_kqgpy_e12 ) \
  python scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPeT2.json $HAD > kutuk/ftm_harman.log 2>&1 || dur "harman"
# ---------------------------------------------------------------- 3. ADS-B düzeltici
if [ "$DUZ_NN" = "1" ]; then EK=_42n; M="model=lgb+nn nn_w=0.7 nn_epok=8"; else EK=_42o; M=""; fi
if [ "$PLANLI" = "1" ]; then EK=${EK/_42n/_42p}; EK=${EK/_42o/_42op}; M="planli=1 $M"; fi
if [ "$V9" = "1" ] && [ "$EK" = "_42p" ]; then EK=_42p9; M="planli=1 model=lgb+nn nn_w=0.9 nn_epok=8 tohumlar=42,2026,7,1,2 bilesen=1"; fi
test -f $P/TEST_${HAD}_adsb$EK.parquet || {  # 11.10: varsa yeniden üretilmez (sıfırdan koşuda klasör boş)
  PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=$HAR PRC_EPOCH_TAM=1 \
    python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF $M ek=$EK 2>&1 | grep -v -i "$F" > kutuk/ftm_duz.log
  [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici"; }
AD=${HAD}_adsb$EK
if [ "${FT_T1:-1}" = "1" ]; then  # 10.10: LIRF yankı hedge'inin p_kilit'i final E1'den (LB v351 − v349 = −0,081; scripts/hedge_e1_degistir.py)
  PRC_HARMAN=$HAR python scripts/hedge_e1_degistir.py $AD ${AD}T > kutuk/ftm_t1.log 2>&1 || dur "T1"
  AD=${AD}T
fi
echo "düzeltici tamam ($AD) $(date +%T)"
# ---------------------------------------------------------------- 4. Model B μ'ye TabM + 8 havalimanlı yerleştirme
NN="durum=park,ktah,kom,gs,pc,yte epok=12"
K=TEST_B_mu_nn_dparkktahkomgspcyte_e12
test -f $P/${K}_t2.parquet || python -u scripts/b_mu_nn.py uretim $NN tohum=42,7 > kutuk/ftm_b_mu_nn.log 2>&1 || dur "B μ TabM"
BN=N; NNF=${K}_t2.parquet
if [ "$BMU5" = "1" ]; then  # 11.10: B μ TabM 5 tohum (0,4 · t2 + 0,6 · t3)
  test -f $P/${K}_t3.parquet || python -u scripts/b_mu_nn.py uretim $NN tohum=2026,1,2 > kutuk/ftm_b_mu_nn_t3.log 2>&1 || dur "B μ TabM t3"
  python - "$K" <<'PY' || dur "B μ ortalaması"
import sys, pandas as pd
P, K = "data_final/preds/", sys.argv[1]
a = pd.read_parquet(P + f"{K}_t2.parquet"); b = pd.read_parquet(P + f"{K}_t3.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
a["bilesen_mu_nn"] = 0.4 * a.bilesen_mu_nn.to_numpy(float) + 0.6 * b.bilesen_mu_nn.to_numpy(float)
a.to_parquet(P + f"{K}_t5.parquet", index=False)
PY
  BN=N5; NNF=${K}_t5.parquet
fi
python scripts/b_mu_nn_yerlestir.py $P tum_dpdcb6c5 tum_dpdcb6c5$BN nn=$NNF > kutuk/ftm_b_yerlestir.log 2>&1 || dur "B μ yerleştirme"
LT=tum6M_adsb_lirfte_mu2_tk; SUF=${BN/N/N8}
if [ "$LIRF" = "1" ]; then  # 11.10: LIRF μ'ye kısmi TabM (B μ TabM 5 tohum, LIRF dahil) + 24 sa karışımı yeniden
  test -f $P/${K}_t5_hepsi.parquet || python -u scripts/b_mu_nn.py uretim hepsi $NN tohum=42,7,2026,1,2 > kutuk/ftm_b_mu_nn_hepsi.log 2>&1 || dur "B μ TabM hepsi"
  python scripts/b_lirf_mu_nn_uygula.py > kutuk/ftm_lirf_mu.log 2>&1 || dur "LIRF μ"
  python -u scripts/b_lirf_tarih_kaymasi.py uretim tum6M_adsb_lirfte_mu2n _tk > kutuk/ftm_lirf_tk.log 2>&1 || dur "LIRF 24 sa"
  LT=tum6M_adsb_lirfte_mu2n_tk; SUF=${SUF}L
fi
python - "$BN" "$LT" "$SUF" <<'PY' || dur "8 havalimanlı yerleştirme"
import sys, pandas as pd
P = "data_final/preds/"
BN, LT, SUF = sys.argv[1:4]
SEKIZ = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM", "EHAM", "LSZH"]
d = pd.read_parquet(P + f"TEST_B_tum_dpdcb6c5{BN}.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
for ad, kay in (("tum6M_adsb", "tum6M_adsb"), ("tum6M_adsb_lirfte", LT)):
    k = pd.read_parquet(P + f"TEST_B_{kay}.parquet")
    if "ADEP_mvt" not in k:
        k = k.merge(pd.read_parquet(P + "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt")
    k = k.merge(d, on="MVT_ID_mvt", how="left")
    m = k.ADEP_mvt.isin(SEKIZ).to_numpy()
    assert k.loc[m, "yeni"].notna().all()
    for c in ("taban", "taksi"):
        if c in k:
            k[c] = k[c].astype(float)
            k.loc[m, c] = k.loc[m, "yeni"]
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_b{SUF}.parquet", index=False)
    print(ad, f"→ _b{SUF}:", int(m.sum()), "satır")
PY
# ---------------------------------------------------------------- 5. B5 (Model B düzelticisi)
BAD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_b${SUF}${B5:+n}
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 $B5 \
  kaynak=tum6M_adsb_b${SUF} guncel=tum6M_adsb_lirfte_b${SUF} ad=$BAD 2>&1 | grep -v -i "$F" > kutuk/ftm_b5.log
test -f $P/TEST_B_$BAD.parquet || dur "B5"
echo "B5 tamam ($BAD) $(date +%T)"
# ---------------------------------------------------------------- 6. gönderim + Y1
python scripts/g_gonderim.py $AD 1.0 b=$BAD lobt=1 "FINAL (4 ay): A = $AD (TabM'li harman + düzeltici), B = $BAD (8 havalimanı, B μ'ye TabM). Ara dosya (Y1'siz)." \
  > kutuk/ftm_g.log 2>&1 || dur "gönderim"
ARA=$(yol kutuk/ftm_g.log); [ -n "$ARA" ] || dur "ara dosya"
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist (1σ) + LOBT. FINAL yapısı (final_tabm.sh)." > kutuk/ftm_k.log 2>&1 || dur "Y1"
SON=$(grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" kutuk/ftm_k.log | grep -v "$(basename $ARA)" | head -1 | tr '\\' '/')
if [ "${FT_YETIM:-1}" = "1" ]; then  # 11.10: yetim B penceresi (LIRF / LTFM; varış kaydından LOBT; scripts/b_yetim_pencere.py)
  python scripts/b_yetim_pencere.py uretim "$SON" "$(basename $SON) + yetim B penceresi. FINAL yapısı (final_tabm.sh)." > kutuk/ftm_yetim.log 2>&1 || dur "yetim"
  K1=$SON; SON=$(grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" kutuk/ftm_yetim.log | grep -v "$(basename $K1)" | head -1 | tr '\\' '/')
  [ -n "$SON" ] || dur "yetim çıktısı"
fi
if [ "${FT_GUNLUK:-1}" = "1" ]; then  # 11.10: günlük kalkış öncesi gecikme kısıtı (yalnız A, LIRF hariç, ±300 sn; scripts/g_gunluk.py)
  python scripts/g_gunluk.py "$SON" a_dosya=$AD haric=LIRF "$(basename $SON) + günlük kalkış öncesi gecikme kısıtı. FINAL yapısı (final_tabm.sh)." \
    > kutuk/ftm_gunluk.log 2>&1 || dur "günlük kısıt"
  K1=$SON; SON=$(grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" kutuk/ftm_gunluk.log | grep -v "$(basename $K1)" | head -1 | tr '\\' '/')
  [ -n "$SON" ] || dur "günlük kısıt çıktısı"
fi
python - "$SON" <<'PY' || dur "denetim"
import sys, pandas as pd
b = pd.read_parquet(sys.argv[1])
assert len(b) == 670_790 and b.MVT_ID_mvt.is_unique and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
print(f"denetim tamam: {len(b):,} satır, ort {b.TAXITIME_SEC_mvt.mean():.1f}")
PY
echo "final aday: ara $ARA → $SON MD5: \`$(md5sum "$SON" | cut -c1-32)\`"
echo "zincir bitti $(date +%T)"

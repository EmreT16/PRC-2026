#!/bin/bash
# 11.10: 10.10 gece kabul edilen kural (b_lirf_mu_esik.py; t 0,95, w 0,6)
# üretimde. (1) B μ TabM'i LIRF dahil tüm test B satırlarına 5 tohumla (b_mu_nn.py uretim hepsi; normal + final),
# (2) scripts/b_lirf_mu_nn_uygula.py → TEST_B_tum6M_adsb_lirfte_mu2n, (3) adaylar (tek fark LIRF μ): normal v361'in
# (A _42p5T, B _dpdcb6c5N85 → _dpdcb6c5N85L), final a64'ün karşılığı (24 sa karışımı _tk finalde yeniden).
# Düzeltici v zincirinden sonra (ekran kartı). Kayıt: kutuk/zincir_lirf_mu_nn.log (+ kutuk/lmn_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
until grep -q "zincir bitti\|HATA" kutuk/duz_v_tohum_cv.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
NN="hepsi durum=park,ktah,kom,gs,pc,yte epok=12 tohum=42,7,2026,1,2"
K=TEST_B_mu_nn_dparkktahkomgspcyte_e12_t5_hepsi
test -f data/preds/$K.parquet || python -u scripts/b_mu_nn.py uretim $NN > kutuk/lmn_tabm_normal.log 2>&1 || dur "TabM μ normal"
echo "TabM μ normal tamam $(date +%T)"
test -f data_final/preds/$K.parquet || PRC_FINAL=1 python -u scripts/b_mu_nn.py uretim $NN > kutuk/lmn_tabm_final.log 2>&1 || dur "TabM μ final"
echo "TabM μ final tamam $(date +%T)"
# ---------------------------------------------------------------- normal: v361 + LIRF μ kısmi TabM
python scripts/b_lirf_mu_nn_uygula.py > kutuk/lmn_uygula_normal.log 2>&1 || dur "uygulama normal"
cat kutuk/lmn_uygula_normal.log
cp data/preds/TEST_B_tum_dpdcb6c5N85.parquet data/preds/TEST_B_tum_dpdcb6c5N85L.parquet
PRC_LIRFTE=tum6M_adsb_lirfte_mu2n PRC_A_AD=D_kqgpPeT2_adsb_42p5T PRC_B_APT=EDDM,EGLL,LEBL,LEMD,LFPG,LTFM,EHAM,LSZH \
  bash scripts/zincir_b_durum2_uretim.sh 1 _dpdcb6c5N85L > kutuk/lmn_normal.log 2>&1
grep -q "zincir bitti" kutuk/lmn_normal.log || dur "normal B zinciri"
SON=$(grep -o "aday: .*" kutuk/lmn_normal.log); echo "normal $SON"
python scripts/lirf_aday_kiyas.py normal "$(echo $SON | grep -o 'strong-moose_v[0-9]*\.parquet' | tail -1)" || dur "normal kıyas"
# ---------------------------------------------------------------- final: a64 + LIRF μ kısmi TabM
export PRC_FINAL=1
python scripts/b_lirf_mu_nn_uygula.py > kutuk/lmn_uygula_final.log 2>&1 || dur "uygulama final"
cat kutuk/lmn_uygula_final.log
python -u scripts/b_lirf_tarih_kaymasi.py uretim tum6M_adsb_lirfte_mu2n _tk > kutuk/lmn_tk_final.log 2>&1 || dur "tarih kayması final"
python scripts/lirf_aday_kiyas.py yerlestir || dur "8 havalimanlı yerleştirme final"
BAD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN85L
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 \
  kaynak=tum6M_adsb_bN85L guncel=tum6M_adsb_lirfte_bN85L ad=$BAD 2>&1 | grep -v -i "$F" > kutuk/lmn_b5_final.log
test -f data_final/preds/TEST_B_$BAD.parquet || dur "B5 final"
python scripts/g_gonderim.py D_kqgpPeT2_adsb_42p5T 1.0 b=$BAD lobt=1 "FINAL (4 ay): a64 + LIRF μ'ye kısmi TabM (p0 > 0,95, w 0,6). Ara dosya (Y1'siz)." > kutuk/lmn_g_final.log 2>&1 || dur "gönderim final"
ARA=gonderim_final/$(grep -o "strong-moose_final_a[0-9]*\.parquet" kutuk/lmn_g_final.log | head -1)
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist (1σ) + LOBT. FINAL: a64 + LIRF μ kısmi TabM." > kutuk/lmn_k_final.log 2>&1 || dur "Y1 final"
SONF=gonderim_final/$(grep -o "strong-moose_final_a[0-9]*\.parquet" kutuk/lmn_k_final.log | grep -v "$(basename $ARA)" | head -1)
echo "final: ara $ARA → $SONF $(grep -o 'MD5: `[0-9a-f]*`' kutuk/lmn_k_final.log | head -1)"
python scripts/lirf_aday_kiyas.py final "$SONF" || dur "final kıyas"
echo "zincir bitti $(date +%T)"

#!/bin/bash
# 06.10: v224'ün MVT_DAY'siz karşılığı. Test tarafındaki 5 bileşen final yapılandırmasıyla
# (PRC_EKOZ_DOSYA=ekoz_secim_final.json: komşu + K2 + yankı + günlük + pist; aynı tohumlar) MVT_DAY olmadan yeniden eğitilir;
# OOF tarafı, ağırlıklar (eski kqgpT), düzeltici ayarları, B dosyası, yankı hedge'i (eski E1), LOBT ve Y1 v224 ile aynı.
# v224 ile farkı = yalnız MVT_DAY. ekoz_secim.json'a dokunulmaz. Hata olursa durur (set -e, çıkış kodları, dosya adı denetimi).
# Kayıt: kutuk/zincir_gunsuz.log, ayrıntılar kutuk/gs_*.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
echo "başladı $(date +%T)"
rm -f kutuk/gs_cpu.done kutuk/gs_gpu.done
EGIT="PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_final.json PRC_GUNSUZ=1"
set +e
(
  env $EGIT bash -c '
    run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
    run gs_lgb scripts/d_tam_egitim.py v_lgb_huber1000_6t_kqgpy_gunsuz && run gs_e1 scripts/d_tam_egitim.py v_e1_huber1000_6t_kqgpy_gunsuz'
  echo "islemci: $? $(date +%T)" > kutuk/gs_cpu.done
) &
(
  env $EGIT bash -c '
    run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
    run gs_xgb scripts/d_tam_egitim.py v_xgb_kqgpy_1500_gunsuz \
    && run gs_dw scripts/d_global_cb.py 9000 dw_lr03_ek5_l2_10_gunsuz_kqgpy 42,2026,7,123,999 lr=0.03 depth=9 kat=ek5 \
         cb:grow_policy=Depthwise cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
    && run gs_gcb scripts/d_global_cb.py 10000 it12000_gunsuz_kqgpy 42,2026,7,123'
  echo "ekran: $? $(date +%T)" > kutuk/gs_gpu.done
) &
wait
set -e
cat kutuk/gs_cpu.done kutuk/gs_gpu.done
grep -q "islemci: 0" kutuk/gs_cpu.done && grep -q "ekran: 0" kutuk/gs_gpu.done || { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
for f in TEST_v_lgb_huber1000_6t_kqgpy_gunsuz TEST_v_e1_huber1000_6t_kqgpy_gunsuz TEST_v_xgb_kqgpy_1500_gunsuz; do
  test -f data/preds/$f.parquet || { echo "eksik: $f; durdu"; exit 1; }
done
T=dw_lr03_ek5_l2_10
run gs_dw_ort5 scripts/gcb_ortalama.py ${T}_ort5_gunsuz_kqgpy ${T}_gunsuz_kqgpy_t42 ${T}_gunsuz_kqgpy_t2026 ${T}_gunsuz_kqgpy_t7 \
  ${T}_gunsuz_kqgpy_t123 ${T}_gunsuz_kqgpy_t999
run gs_gcb_ort4 scripts/gcb_ortalama.py it12000_ort4_gunsuz_kqgpy it12000_gunsuz_kqgpy_t42 it12000_gunsuz_kqgpy_t2026 \
  it12000_gunsuz_kqgpy_t7 it12000_gunsuz_kqgpy_t123
export PRC_HARMAN=kqgpGe PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
run gs_harman scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpGe.json D_kqgpGe
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
run gs_duz scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_40a
test -f data/preds/TEST_D_kqgpGe_adsb_40a.parquet || { echo "düzeltici çıktısı yok; durdu"; exit 1; }
python scripts/g_gonderim.py D_kqgpGe_adsb_40a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "v223'ün MVT_DAY'siz karşılığı: test tarafındaki 5 bileşen final yapılandırmasıyla (ekoz_secim_final.json, aynı tohumlar) MVT_DAY olmadan; ağırlık, düzeltici, B, hedge, LOBT aynı. Ara dosya; kalibre edilmiş hali gönderilir." > kutuk/gs_gonderim.log 2>&1
ARA=$(grep -o "gonderim[\\/]strong-moose_v[0-9]*\.parquet" kutuk/gs_gonderim.log | head -1 | tr '\\' '/')
test -n "$ARA" && test -f "$ARA" || { echo "ara gönderim dosyası bulunamadı; durdu"; exit 1; }
echo "ara: $ARA"
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v224 ile farkı = yalnız MVT_DAY (Model A test bileşenlerinde takvim günü ezberi)." > kutuk/gs_kalibre.log 2>&1
SON=$(grep -o "gonderim[\\/]strong-moose_v[0-9]*\.parquet" kutuk/gs_kalibre.log | head -1 | tr '\\' '/')
test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası bulunamadı; durdu"; exit 1; }
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v224.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"{sys.argv[1]} − v224: değişen {int((np.abs(d) > 1e-6).sum()):,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.2f}")
PY
echo "zincir bitti $(date +%T)"

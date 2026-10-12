#!/bin/bash
# 08.10: Model A ADS-B düzelticisine ikinci model ailesi (model=lgb+cb: LightGBM + CatBoost Huber δ 600, 800 yineleme,
# derinlik 6; iki ailenin ortalaması). Model B μ'de LightGBM + CatBoost 769,8 → 756,8 verdi; düzelticide hiç denenmemişti
# (cb yolu kategorik 'tip' sütununda hatalıydı, 08.10'da düzeltildi). Adımlar:
#  1) CV Oca/Tem (karşılık kutuk/dcv_duz_parkb.log: 196,14; Oca 172,49, Tem 213,05). Kapı: ağırlıklı ≤ 196,09 ve iki ay da iyi.
#  2) Doğrulama + H1 / G2 OOF'u: sinama=3,10 (karşılık dcv_duz_bosluk_parkb) ve 11,12 (dcv_duz_parkb_kas).
#  3) Üretim _43o (3 tohum × 2 aile) → H1 (PRC_DUZ_CV=lgbcb) → G2 (ha=1) → gönderim: v297 ile aynı B (bdpdcb).
# Kayıt: kutuk/zincir_duz_lgbcb.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1 parkb=1 model=lgb+cb"
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
F='warn\|X\["tip"\]'
echo "başladı $(date +%T)"
if [ ! -f data/preds/ADSB_duz_cv_lgbcb.parquet ]; then
  python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -v -i "$F" > kutuk/dcv_duz_lgbcb.log
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_lgbcb.parquet
fi
echo "CV Oca/Tem tamam $(date +%T)"; grep "k=1.0 " kutuk/dcv_duz_parkb.log kutuk/dcv_duz_lgbcb.log
python - <<'PY' || { echo "KAPI: kazanç yok, zincir durdu $(date +%T)"; exit 0; }
import re, sys
def k1(f):
    for s in open(f, encoding="utf-8"):
        if s.startswith("k=1.0 "):
            return [float(x) for x in s.split()[1:]]
a, b = k1("kutuk/dcv_duz_parkb.log"), k1("kutuk/dcv_duz_lgbcb.log")
print(f"Oca {a[0]} → {b[0]}, Tem {a[2]} → {b[2]}, ağırlıklı {a[4]} → {b[4]}")
sys.exit(0 if (b[4] <= a[4] - 0.05 and b[0] < a[0] and b[2] < a[2]) else 1)
PY
echo "KAPI geçti"
if [ ! -f data/preds/ADSB_duz_cv_bosluk_lgbcb.parquet ]; then
  python -u scripts/adsb_duzeltici.py cv $D sinama=3,10 2>&1 | grep -v -i "$F" > kutuk/dcv_duz_bosluk_lgbcb.log
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_bosluk_lgbcb.parquet
fi
if [ ! -f data/preds/ADSB_duz_cv_lgbcb_kas.parquet ]; then
  python -u scripts/adsb_duzeltici.py cv $D sinama=11,12 2>&1 | grep -v -i "$F" > kutuk/dcv_duz_lgbcb_kas.log
  cp data/preds/ADSB_duz_cv_h0_huber.parquet data/preds/ADSB_duz_cv_lgbcb_kas.parquet
fi
echo "doğrulama tamam $(date +%T)"
grep "k=1.0 " kutuk/dcv_duz_bosluk_parkb.log kutuk/dcv_duz_bosluk_lgbcb.log kutuk/dcv_duz_parkb_kas.log kutuk/dcv_duz_lgbcb_kas.log
test -f data/preds/TEST_D_kqgpPe_adsb_43o.parquet || PRC_EPOCH_TAM=1 python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_43o 2>&1 | grep -v -i "$F" > kutuk/adsb_uretim_D_kqgpPe_43o.log
test -f data/preds/TEST_D_kqgpPe_adsb_43o.parquet || { echo "üretim _43o YOK"; exit 1; }
echo "üretim _43o tamam $(date +%T)"
PRC_DUZ_CV=lgbcb python -u scripts/havalimani_agirlik.py D_kqgpPe_adsb_43o 2>&1 | grep -v -i warn > kutuk/h1_lgbcb.log
PRC_DUZ_CV=lgbcb python -u scripts/gecikme_asama2.py D_kqgpPe_adsb_43o_ha ha=1 2>&1 | grep -v -i warn > kutuk/g2_lgbcb.log
grep "toplam" kutuk/h1_lgbcb.log kutuk/g2_lgbcb.log
test -f data/preds/TEST_D_kqgpPe_adsb_43o_ha_g2.parquet || { echo "H1 / G2 çıktısı yok"; exit 1; }
B=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb
python scripts/g_gonderim.py D_kqgpPe_adsb_43o_ha_g2 1.0 b=$B lobt=1 "v297 ile aynı B; Model A düzelticisi LightGBM + CatBoost (_43o, model=lgb+cb) + H1 + G2 (lgbcb OOF'larıyla). Ara dosya." > kutuk/lgbcb_gonderim.log 2>&1
ARA=$(yol kutuk/lgbcb_gonderim.log)
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v297 ile farkı = yalnız Model A düzelticisi (lgb+cb) ve onunla yeniden eğitilen H1 / G2." > kutuk/lgbcb_kalibre.log 2>&1
SON=$(yol kutuk/lgbcb_kalibre.log)
echo "aday: ara $ARA → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/lgbcb_kalibre.log | head -1)"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v297.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"   − v297: değişen {(np.abs(d) > 1e-3).sum():,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, |fark| > 300 sn {int((np.abs(d) > 300).sum())}")
PY
echo "zincir bitti $(date +%T)"

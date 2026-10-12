#!/bin/bash
# 07.10 gece: v247 (B5 12 ay) + B5 düzeltme katsayısı k = 1,25 (CV'ye göre: kutuk/dcv_b5_12ay_d600.log, LIRF dışı 8 havalimanının
# 7'sinde ve 6 ayın 5'inde k = 1'den iyi; LB'ye göre seçilmedi). v247 ile farkı = yalnız k. Kayıt: kutuk/zincir_b5_k125.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
AD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125
PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 ad=$AD > kutuk/b5g_k125_uretim.log 2>&1
test -f data/preds/TEST_B_$AD.parquet || { echo "B5 çıktısı yok; durdu"; exit 1; }
python scripts/g_gonderim.py D_kqgpPe_adsb_37a 1.0 b=$AD lobt=1 "v246 ile aynı (B5 12 ay), B5 düzeltme katsayısı k = 1,25 (CV'ye göre). Ara dosya." > kutuk/b5g_k125_gonderim.log 2>&1
ARA=$(grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" kutuk/b5g_k125_gonderim.log | head -1 | tr '\' '/')
test -n "$ARA" && test -f "$ARA" || { echo "ara dosya yok; durdu"; exit 1; }
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v247 ile farkı = yalnız B5 k = 1,25 (CV'ye göre seçildi)." > kutuk/b5g_k125_kalibre.log 2>&1
SON=$(grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" kutuk/b5g_k125_kalibre.log | head -1 | tr '\' '/')
test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası yok; durdu"; exit 1; }
echo "ara $ARA, aday $SON"; grep MD5 kutuk/b5g_k125_kalibre.log
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
for ref in ("gonderim/strong-moose_v247.parquet", "gonderim/strong-moose_v243.parquet"):
    a = pd.read_parquet(ref); b = pd.read_parquet(sys.argv[1])
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  − {ref}: değişen {int((np.abs(d) > 1e-6).sum()):,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.3f}")
PY
echo "zincir bitti $(date +%T)"

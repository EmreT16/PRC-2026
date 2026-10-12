#!/bin/bash
# 07.10: B5'e OPDI. v257 (Model A düzelticisi _40o) + B5: 12 ay, k = 1,25 (v249'daki
# gibi; CV'ye göre), iz5, nmsiz, damga5 + opdi=1 opdi_satir=1 + PRC_EPOCH_TAM=1.
# CV (kutuk/dcv_b5_12ay_opdi*.log): ortak 7.015 izli B satırında LIRF hariç ≈ nötr, OPDI ile eklenen 111 satır 355,7 → 265,7.
# Kayıt: kutuk/zincir_opdi_b5_uretim.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
AD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi
echo "başladı $(date +%T)"
PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 ad=$AD > kutuk/b5o_uretim.log 2>&1
test -f data/preds/TEST_B_$AD.parquet || { echo "B5 çıktısı yok; durdu"; exit 1; }
grep -E "test B satırı" kutuk/b5o_uretim.log
python scripts/g_gonderim.py D_kqgpPe_adsb_40o 1.0 b=$AD lobt=1 "v257 ile aynı (Model A düzelticisi _40o) + B5: 12 ay, k = 1,25, OPDI (özellik + izsiz satırlar), mvt_epoch 12 ay. Ara dosya." > kutuk/b5o_gonderim.log 2>&1
ARA=$(yol kutuk/b5o_gonderim.log); test -n "$ARA" && test -f "$ARA" || { echo "ara dosya yok; durdu"; exit 1; }
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v257 ile farkı = yalnız B5 (12 ay, k = 1,25, OPDI, epoch)." > kutuk/b5o_kalibre.log 2>&1
SON=$(yol kutuk/b5o_kalibre.log); test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası yok; durdu"; exit 1; }
echo "ara $ARA, aday $SON"; grep MD5 kutuk/b5o_kalibre.log
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1])
for ref in ("gonderim/strong-moose_v257.parquet", "gonderim/strong-moose_v249.parquet"):
    a = pd.read_parquet(ref); assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  − {ref}: değişen {int((np.abs(d) > 1e-6).sum()):,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.3f}")
PY
echo "zincir bitti $(date +%T)"

#!/bin/bash
# 07.10 akşam: v259 + B5'te B satırlarının komşu sapması / merkezleme final Model A test tabanıyla (PRC_HARMAN=kqgpPe; önceden
# kqgpxg). kqgpxg ile kqgpPe'nin OOF bileşenleri ve ağırlıkları aynı → eğitim aynı, yalnız test A çerçevesi değişir.
# v259 ile tek fark bu (v245'in v243 üstündeki karşılığı). Kayıt: kutuk/zincir_b5_pe2.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
AD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_pe
echo "başladı $(date +%T)"
PRC_HARMAN=kqgpPe PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 ad=$AD > kutuk/b5pe2_uretim.log 2>&1
test -f data/preds/TEST_B_$AD.parquet || { echo "B5 çıktısı yok; durdu"; exit 1; }
python scripts/g_gonderim.py D_kqgpPe_adsb_40o 1.0 b=$AD lobt=1 "v259 ile aynı, B5'te B satırlarının komşu sapması final Model A tabanıyla (kqgpPe). Ara dosya." > kutuk/b5pe2_gonderim.log 2>&1
ARA=$(yol kutuk/b5pe2_gonderim.log); test -n "$ARA" && test -f "$ARA" || { echo "ara dosya yok; durdu"; exit 1; }
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v259 ile farkı = yalnız B5 komşu tabanı (kqgpxg → kqgpPe)." > kutuk/b5pe2_kalibre.log 2>&1
SON=$(yol kutuk/b5pe2_kalibre.log); test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası yok; durdu"; exit 1; }
echo "ara $ARA, aday $SON"; grep MD5 kutuk/b5pe2_kalibre.log
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v259.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
r = pd.read_parquet("../../PRC_2/data_2/processed/dep_rank.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC"])
B = a[["MVT_ID_mvt"]].merge(r, on="MVT_ID_mvt", how="left")["NM_TAXI_DIFF_SEC"].isna().to_numpy()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"  − v259: değişen {int((np.abs(d) > 1e-6).sum()):,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn | B satırları: değişen {int((np.abs(d[B]) > 1e-6).sum()):,}, RMS {np.sqrt(np.mean(d[B] ** 2)):.1f} | A satırları RMS {np.sqrt(np.mean(d[~B] ** 2)):.2f}")
PY
echo "zincir bitti $(date +%T)"

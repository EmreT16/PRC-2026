#!/bin/bash
# 07.10: OPDI'li Model A ADS-B düzelticisiyle aday. v243 ile tek fark Model A düzelticisi
# (_37a → $1); B tarafı v243'ünkü (D_kqgpxg_17a_b5_iz5_nmsiz_d5). Ayarlar _37a ile aynı (scripts/zincir_yedek_eski_agirlik.sh).
# Kullanım: bash scripts/zincir_opdi_uretim.sh <ek> <epoch 0|1> "<not>" <ek seçenekler...>
#   ör. bash scripts/zincir_opdi_uretim.sh _39o 0 "OPDI satırları" opdi=1 opdi_satir=1
# Kayıt: kutuk/zincir_opdi_uretim<ek>.log
set -e
cd "$(dirname "$0")/.."
EK=$1; EP=$2; NOT=$3; shift 3
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
[ "$EP" = "1" ] && export PRC_EPOCH_TAM=1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
echo "başladı $(date +%T) | $EK epoch=$EP | $*"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=$EK "$@" 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpPe$EK.log
test -f data/preds/TEST_D_kqgpPe_adsb$EK.parquet || { echo "düzeltici çıktısı yok; durdu"; exit 1; }
tail -14 kutuk/adsb_uretim_D_kqgpPe$EK.log
unset PRC_HARMAN PRC_EPOCH_TAM
python scripts/g_gonderim.py D_kqgpPe_adsb$EK 1.0 b=D_kqgpxg_17a_b5_iz5_nmsiz_d5 lobt=1 "v243 ile aynı, Model A ADS-B düzelticisi $EK: $NOT. Ara dosya." > kutuk/opdi_gonderim$EK.log 2>&1
ARA=$(yol kutuk/opdi_gonderim$EK.log)
test -n "$ARA" && test -f "$ARA" || { echo "ara dosya yok; durdu"; exit 1; }
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v243 ile farkı = yalnız Model A düzelticisi ($NOT)." > kutuk/opdi_kalibre$EK.log 2>&1
SON=$(yol kutuk/opdi_kalibre$EK.log)
test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası yok; durdu"; exit 1; }
echo "ara $ARA, aday $SON"; grep MD5 kutuk/opdi_kalibre$EK.log
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v243.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
r = pd.read_parquet("../../PRC_2/data_2/processed/dep_rank.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
x = a[["MVT_ID_mvt"]].merge(r, on="MVT_ID_mvt", how="left")
x["d"] = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"− v243: değişen {int((x.d.abs() > 1e-6).sum()):,} satır, RMS {np.sqrt(np.mean(x.d ** 2)):.2f} sn, ort {x.d.mean():+.3f}")
print(x.groupby(["ADEP_mvt", "MVT_MONTH"]).d.agg(["mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(2).unstack().to_string())
PY
echo "zincir bitti $(date +%T)"

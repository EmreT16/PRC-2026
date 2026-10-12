#!/bin/bash
# 07.10 gece: v243'ün B5'i üzerine iki aday, ikisi de v243'e karşı tek değişkenli.
#  1) CV: B5'i 12 ayla eğitmek (PRC_DUZ_EK=split_4,split_5,split_6; karşılığı kutuk/dcv_b5_iz5_nmsiz_d5.log, izli B 2.076,9)
#  2) Aday "pe": B satırlarının komşu sapması / merkezleme final Model A test tabanıyla (PRC_HARMAN=kqgpPe; A eğitim verisi aynı:
#     kqgpxg ile kqgpPe'nin OOF bileşenleri ve ağırlıkları birebir aynı, yalnız test A çerçevesi değişir)
#  3) Aday "12ay": B5 12 ayla (PRC_HARMAN=kqgpxg). Hangisinin yükleneceği CV'ye göre (12 ay CV'de kötüyse yüklenmez).
# Kayıt: kutuk/zincir_b5_gece.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
B5="apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 iz5=1 nmsiz=1 damga5=1"
echo "başladı $(date +%T)"
PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 python -u scripts/adsb_b_duzeltici5.py cv iz5=1 nmsiz=1 damga5=1 > kutuk/dcv_b5_d5_12ay.log 2>&1
echo "CV 12 ay bitti $(date +%T)"; grep -A3 "izli B" kutuk/dcv_b5_d5_12ay.log | head -4

aday() {  # $1 = ad eki, $2 = PRC_HARMAN, $3 = PRC_DUZ_EK, $4 = not
  local AD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_$1
  PRC_HARMAN=$2 PRC_DUZ_EK=$3 python -u scripts/adsb_b_duzeltici5.py uretim $B5 ad=$AD > kutuk/b5g_$1_uretim.log 2>&1
  test -f data/preds/TEST_B_$AD.parquet || { echo "B5 çıktısı yok ($1); durdu"; exit 1; }
  python scripts/g_gonderim.py D_kqgpPe_adsb_37a 1.0 b=$AD lobt=1 "$4 Ara dosya." > kutuk/b5g_$1_gonderim.log 2>&1
  local ARA=$(grep -o "gonderim[\\/]strong-moose_v[0-9]*\.parquet" kutuk/b5g_$1_gonderim.log | head -1 | tr '\\' '/')
  test -n "$ARA" && test -f "$ARA" || { echo "ara gönderim dosyası bulunamadı ($1); durdu"; exit 1; }
  python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. $4" > kutuk/b5g_$1_kalibre.log 2>&1
  local SON=$(grep -o "gonderim[\\/]strong-moose_v[0-9]*\.parquet" kutuk/b5g_$1_kalibre.log | head -1 | tr '\\' '/')
  test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası bulunamadı ($1); durdu"; exit 1; }
  echo "[$1] ara $ARA, aday $SON"; grep -E "MD5" kutuk/b5g_$1_kalibre.log
  python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v243.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"  − v243: değişen {int((np.abs(d) > 1e-6).sum()):,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.3f}")
PY
}
aday pe kqgpPe "" "v243 ile aynı, B5'te B satırlarının komşu sapması / merkezleme final Model A test tabanıyla (kqgpPe; önceden kqgpxg). v243 ile farkı = yalnız bu."
aday 12ay kqgpxg split_4,split_5,split_6 "v243 ile aynı, B5 12 ayla eğitildi (önceden 6 ay). v243 ile farkı = yalnız bu."
echo "zincir bitti $(date +%T)"

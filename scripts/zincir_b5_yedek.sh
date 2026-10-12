#!/bin/bash
# 07.10 gece: yarının yedek / birleşik B5 dosyaları (hepsi v243'ün Model A tarafıyla; yalnız B5 değişir).
#  y6k  : B5 6 ay + k = 1,25 (v247 kötü çıkarsa v249 yerine)
#  12pe : B5 12 ay + k = 1,25 + komşu sapması final Model A tabanıyla (kqgpPe) = v249 + v245 birleşimi
# İlk koşuda dosya yolu dönüşümü (sed) hata verdi; y6k'nın ara dosyası (v250) oluşmuştu. devam=1 → y6k yalnız kalibre edilir.
# Kayıt: kutuk/zincir_b5_yedek.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
yol() { grep -o "gonderim[\\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\\' '/'; }
kalibre() {  # $1 ek, $2 ara dosya, $3 not
  python -u scripts/g_kalibre.py "$2" pist=1 "$(basename $2) + Y1 havalimanı + pist + LOBT. $3" > kutuk/b5y_$1_kalibre.log 2>&1
  local SON=$(yol kutuk/b5y_$1_kalibre.log)
  test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası yok ($1); durdu"; exit 1; }
  echo "[$1] ara $2, aday $SON"; grep MD5 kutuk/b5y_$1_kalibre.log
  python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
for ref in ("gonderim/strong-moose_v243.parquet", "gonderim/strong-moose_v249.parquet"):
    a = pd.read_parquet(ref); b = pd.read_parquet(sys.argv[1])
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  − {ref}: RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.3f}")
PY
}
uret() {  # $1 ek, $2 PRC_HARMAN, $3 PRC_DUZ_EK, $4 not
  local AD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_$1
  PRC_HARMAN=$2 PRC_DUZ_EK=$3 python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG \
    k=1.25 iz5=1 nmsiz=1 damga5=1 ad=$AD > kutuk/b5y_$1_uretim.log 2>&1
  test -f data/preds/TEST_B_$AD.parquet || { echo "B5 çıktısı yok ($1); durdu"; exit 1; }
  python scripts/g_gonderim.py D_kqgpPe_adsb_37a 1.0 b=$AD lobt=1 "$4 Ara dosya." > kutuk/b5y_$1_gonderim.log 2>&1
  local ARA=$(yol kutuk/b5y_$1_gonderim.log)
  test -n "$ARA" && test -f "$ARA" || { echo "ara dosya yok ($1); durdu"; exit 1; }
  kalibre $1 "$ARA" "$4"
}
echo "başladı $(date +%T)"
N6="YEDEK: v243 + B5 k = 1,25 (6 ay; CV'ye göre). v247 kötü çıkarsa v249 yerine; v243 ile farkı = yalnız k."
if [ "$1" = "devam=1" ]; then
  kalibre y6k gonderim/strong-moose_v250.parquet "$N6"
else
  uret y6k kqgpxg "" "$N6"
fi
uret 12pe kqgpPe split_4,split_5,split_6 "BİRLEŞİK: v249 (B5 12 ay + k = 1,25) + v245 (komşu sapması final Model A tabanıyla)."
echo "zincir bitti $(date +%T)"

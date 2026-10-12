#!/bin/bash
# 07.10: v239 + B5'te DAMGA5 (park ayraçlı kayıt alışkanlığı) düzeltmesi. DAMGA5 önceden B
# satırlarında hesaplanmadığı için ortak_matris'in sütun kesişiminde A satırlarından da düşüyor, B5'te %100 boş kalıyordu.
# CV: izli B 2.077,5 → 2.076,9, LIRF dışı 6/8 havalimanı iyi (kutuk/dcv_b5_iz5_nmsiz_d5.log). v239 ile farkı = yalnız damga5.
# Kayıt: kutuk/zincir_b5_damga5.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg
python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 iz5=1 nmsiz=1 damga5=1 \
  ad=D_kqgpxg_17a_b5_iz5_nmsiz_d5 > kutuk/b5_iz5_nmsiz_d5_uretim.log 2>&1
unset PRC_HARMAN PRC_ADSB_OZ PRC_ADSB_GEVSEK
test -f data/preds/TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5.parquet || { echo "B5 çıktısı yok; durdu"; exit 1; }
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz.parquet"); b = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5.parquet")
x = a.merge(b[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", suffixes=("", "_y")); x["f"] = x.taksi_y - x.taksi
print("damga5 − nmsiz (B satırları):")
print(x.groupby("ADEP_mvt").f.agg(["size", lambda s: int((s.abs() > 1e-6).sum()), "mean", lambda s: float(np.sqrt((s**2).mean()))]).round(1).to_string())
PY
python scripts/g_gonderim.py D_kqgpPe_adsb_37a 1.0 b=D_kqgpxg_17a_b5_iz5_nmsiz_d5 lobt=1 "v238 ile aynı (B5 bugünkü kod + iz5 + NM'siz) + B5'te DAMGA5 B satırlarında da hesaplanıyor (önceden %100 boştu). Ara dosya." > kutuk/b5d_gonderim.log 2>&1
ARA=$(grep -o "gonderim[\\/]strong-moose_v[0-9]*\.parquet" kutuk/b5d_gonderim.log | head -1 | tr '\\' '/')
test -n "$ARA" && test -f "$ARA" || { echo "ara gönderim dosyası bulunamadı; durdu"; exit 1; }
grep -E "Dosya|MD5" kutuk/b5d_gonderim.log
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v239 ile farkı = yalnız B5'te DAMGA5 düzeltmesi." > kutuk/b5d_kalibre.log 2>&1
SON=$(grep -o "gonderim[\\/]strong-moose_v[0-9]*\.parquet" kutuk/b5d_kalibre.log | head -1 | tr '\\' '/')
test -n "$SON" && test -f "$SON" || { echo "kalibre dosyası bulunamadı; durdu"; exit 1; }
grep -E "Dosya|MD5" kutuk/b5d_kalibre.log
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
for ref in ("gonderim/strong-moose_v239.parquet", "gonderim/strong-moose_v224.parquet"):
    a = pd.read_parquet(ref); b = pd.read_parquet(sys.argv[1])
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"{sys.argv[1]} − {ref}: değişen {int((np.abs(d) > 1e-6).sum()):,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.3f}")
PY
echo "zincir bitti $(date +%T)"

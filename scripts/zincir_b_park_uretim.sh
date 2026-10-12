#!/bin/bash
# 07.10 gece: Model B tüm satırlı modele park yeri dizisi + komşu NM'li kalkışların tahmini taksisi (durum=park,ktah;
# scripts/park_b_ozellik.py). CV (6 havalimanı, 24 sa hariç): 844,7 → 775,2; 4/4 çeyrek, 6/6 havalimanı iyi
# (kutuk/b_park_ktah_cv.log, scripts/b_park_degerlendir.py). Zincir zincir_b_durum_uretim.sh ile aynı: 6 havalimanının
# (EDDM, EGLL, LEBL, LEMD, LFPG, LTFM) tabanı yenisiyle değişir (EHAM / LSZH / EDDF v126, LIRF aynen); B5 v259'daki
# ayarlarla (A satırlarıyla eğitilir, B tabanına göre uygulanır). Gönderimler: v259 + yeni B; v273 (H1 + G2) + yeni B.
# Kayıt: kutuk/zincir_b_park_uretim.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
echo "başladı $(date +%T)"
test -f data/preds/TEST_B_tum_dparkktah.parquet || python -u scripts/b_uretim_tum.py durum=park,ktah ad=tum_dparkktah > kutuk/b_uretim_park.log 2>&1 || { echo "B üretimi hatalı"; exit 1; }
tail -n 6 kutuk/b_uretim_park.log
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
ALTI = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
d = pd.read_parquet(P + "TEST_B_tum_dparkktah.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
for ad in ("tum6M_adsb", "tum6M_adsb_lirfte"):
    k = pd.read_parquet(P + f"TEST_B_{ad}.parquet")
    if "ADEP_mvt" not in k:
        k = k.merge(pd.read_parquet(P + "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt")
    k = k.merge(d, on="MVT_ID_mvt", how="left")
    m = k.ADEP_mvt.isin(ALTI).to_numpy()
    assert k.loc[m, "yeni"].notna().all()
    for c in ("taban", "taksi"):
        if c in k:
            k[c] = k[c].astype(float)
            k.loc[m, c] = k.loc[m, "yeni"]
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_bpark.parquet", index=False)
    print(ad, "→ _bpark:", int(m.sum()), "satır değişti")
PY
AD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bpark
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 \
  kaynak=tum6M_adsb_bpark guncel=tum6M_adsb_lirfte_bpark ad=$AD 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/b5_bpark.log
test -f data/preds/TEST_B_$AD.parquet || { echo "B5 çıktısı yok"; exit 1; }
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi.parquet")
b = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bpark.parquet")
x = a.merge(b[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", suffixes=("", "_y")); x["f"] = x.taksi_y - x.taksi
ALTI = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
print(f"6 havalimanı dışı maks fark (yeniden üretim denetimi): {x.loc[~x.ADEP_mvt.isin(ALTI), 'f'].abs().max():.4f} sn")
print(x.groupby("ADEP_mvt").f.agg(["size", lambda s: int((s.abs() > 1e-6).sum()), "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
PY
for A_AD in "D_kqgpPe_adsb_40o|v259" "D_kqgpPe_adsb_40o_ha_g2|v273"; do
  AA="${A_AD%%|*}"; REF="${A_AD#*|}"
  python scripts/g_gonderim.py $AA 1.0 b=$AD lobt=1 "$REF ile aynı A tarafı + Model B tüm satırlı modele park yeri dizisi / komşu taksi (durum=park,ktah; 6 havalimanı). Ara dosya." > kutuk/bpark_gonderim.log 2>&1
  ARA=$(yol kutuk/bpark_gonderim.log)
  python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. $REF ile farkı = yalnız Model B park yeri / komşu taksi özellikleri." > kutuk/bpark_kalibre.log 2>&1
  SON=$(yol kutuk/bpark_kalibre.log)
  echo "$REF + Bpark: ara $ARA → aday $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/bpark_kalibre.log | head -1)"
  python - "$SON" "$REF" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet(f"gonderim/strong-moose_{sys.argv[2]}.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"   − {sys.argv[2]}: değişen {(np.abs(d)>1e-3).sum():,} satır, RMS {np.sqrt(np.mean(d**2)):.2f} sn, |fark|>300 sn {int((np.abs(d)>300).sum())}")
PY
done
echo "zincir bitti $(date +%T)"

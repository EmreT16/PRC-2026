#!/bin/bash
# 07.10 gece: Model B tüm satırlı model durum=park,ktah + kom,gs,pc,yte (CV 6 havalimanı, 24 sa hariç: 775,2 → 769,4;
# 4/4 çeyrek, 4/6 havalimanı; kutuk/b_park_durum_cv.log). İsteğe bağlı tohum sayısı ($1, varsayılan 1;
# b_uretim_tum.py tohum=N). Üretim v289 ile aynı düzen: 6 havalimanı (EDDM, EGLL, LEBL, LEMD, LFPG, LTFM) yeni tabana,
# EHAM / LSZH / EDDF v126, LIRF aynen; B5 v259 ayarlarıyla; A tarafı D_kqgpPe_adsb_42o_ha_g2 (v289'daki gibi).
# Kayıt: kutuk/zincir_b_durum2_uretim[_t<N>].log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
T=${1:-1}
EK=${2:-_dpd$( [ "$T" -gt 1 ] && echo "_t$T" || true )}  # $2: hazır TEST_B_tum<ek>.parquet (ör. _dpdmu: μ dpd, diğer bileşenler dparkktah)
export PRC_A_AD=${PRC_A_AD:-D_kqgpPe_adsb_42o_ha_g2}  # 08.10: A tarafı (v305: D_kqgpPe_adsb_42o_ha_g2lgbcb_g2t)
export PRC_B_APT=${PRC_B_APT:-EDDM,EGLL,LEBL,LEMD,LFPG,LTFM}  # 08.10: yeni tabana geçen havalimanları (8: + EHAM,LSZH)
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
echo "başladı $(date +%T) (tohum $T, ek $EK)"
test -f data/preds/TEST_B_tum$EK.parquet || python -u scripts/b_uretim_tum.py durum=park,ktah,kom,gs,pc,yte ad=tum$EK tohum=$T 2>&1 | grep -v -i warn > kutuk/b_uretim$EK.log || { echo "B üretimi hatalı"; exit 1; }
test -f data/preds/TEST_B_tum$EK.parquet || { echo "B çıktısı yok"; exit 1; }
tail -n 8 kutuk/b_uretim$EK.log 2>/dev/null || echo "(B üretimi hazır dosyadan: TEST_B_tum$EK.parquet)"
python - "$EK" <<'PY'
import os, sys, pandas as pd, numpy as np
P, EK = "data/preds/", sys.argv[1]
ALTI = os.environ["PRC_B_APT"].split(",")
d = pd.read_parquet(P + f"TEST_B_tum{EK}.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
e = pd.read_parquet(P + "TEST_B_tum_dparkktah.parquet")[["MVT_ID_mvt", "ADEP_mvt", "taksi"]]
x = e.merge(d, on="MVT_ID_mvt"); x["f"] = x.yeni - x.taksi
print("tum_dparkktah'a göre fark (6 havalimanı):")
print(x[x.ADEP_mvt.isin(ALTI)].groupby("ADEP_mvt").f.agg(["size", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
for ad, kay in (("tum6M_adsb", "tum6M_adsb"), ("tum6M_adsb_lirfte", os.environ.get("PRC_LIRFTE", "tum6M_adsb_lirfte"))):
    k = pd.read_parquet(P + f"TEST_B_{kay}.parquet")  # 08.10: PRC_LIRFTE = LIRF düzeni kaynağı (ör. yeni μ'lü)
    if "ADEP_mvt" not in k:
        k = k.merge(pd.read_parquet(P + "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt")
    k = k.merge(d, on="MVT_ID_mvt", how="left")
    m = k.ADEP_mvt.isin(ALTI).to_numpy()
    assert k.loc[m, "yeni"].notna().all()
    for c in ("taban", "taksi"):
        if c in k:
            k[c] = k[c].astype(float)
            k.loc[m, c] = k.loc[m, "yeni"]
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_b{EK[1:]}.parquet", index=False)
    print(ad, f"→ _b{EK[1:]}:", int(m.sum()), "satır değişti")
PY
AD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_b${EK:1}
test -f data/preds/TEST_B_$AD.parquet || PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 \
  kaynak=tum6M_adsb_b${EK:1} guncel=tum6M_adsb_lirfte_b${EK:1} ad=$AD 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/b5_b${EK:1}.log
test -f data/preds/TEST_B_$AD.parquet || { echo "B5 çıktısı yok"; exit 1; }
python - "$AD" <<'PY'
import sys, pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bpark.parquet")
b = pd.read_parquet(P + f"TEST_B_{sys.argv[1]}.parquet")
x = a.merge(b[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", suffixes=("", "_y")); x["f"] = x.taksi_y - x.taksi
print("B5 sonrası, _bpark'a (v289'un B'si) göre fark:")
print(x.groupby("ADEP_mvt").f.agg(["size", lambda s: int((s.abs() > 1e-6).sum()), "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
PY
python scripts/g_gonderim.py $PRC_A_AD 1.0 b=$AD lobt=1 "A tarafı $PRC_A_AD + Model B $EK (durum park,ktah + kom,gs,pc,yte; yeni tabanlı havalimanları $PRC_B_APT). Ara dosya." > kutuk/b${EK:1}_gonderim.log 2>&1
ARA=$(yol kutuk/b${EK:1}_gonderim.log)
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist + LOBT. v289 ile farkı = yalnız Model B ($EK)." > kutuk/b${EK:1}_kalibre.log 2>&1
SON=$(yol kutuk/b${EK:1}_kalibre.log)
echo "aday: ara $ARA → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/b${EK:1}_kalibre.log | head -1)"
python - "$SON" <<'PY'
import sys, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v289.parquet"); b = pd.read_parquet(sys.argv[1])
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"   − v289: değişen {(np.abs(d)>1e-3).sum():,} satır, RMS {np.sqrt(np.mean(d**2)):.2f} sn, |fark|>300 sn {int((np.abs(d)>300).sum())}")
PY
echo "zincir bitti $(date +%T)"

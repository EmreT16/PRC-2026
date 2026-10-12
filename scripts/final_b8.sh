#!/bin/bash
# 09.10 akşam: final modunda 8 havalimanlı Model B (EHAM / LSZH da yeni tabana; Oca / Tem karşılığı v334). CV (B5 öncesi, 24 sa
# hariç): EHAM yeni 310,0 ↔ v126 353,1 (3/4 çeyrek), LSZH 517,7 ↔ 593,6 (4/4); EDDF 315,3 ↔ 315,2 (2/4) → EDDF eski tabanda.
# final_son.sh'nin yerleştirme + B5 + gönderim adımları, ALTI → 8 havalimanı; A değişkenlerinin hepsiyle.
# Kayıt: kutuk/final_b8.log (+ kutuk/fs8_*.log)
cd "$(dirname "$0")/.."  # 09.10: göreli kök (depo / tekrar kopyası)
export PYTHONIOENCODING=utf-8 PRC_FINAL=1 PRC_Y1_ESIK=1.0
F='warn\|X\["tip"\]'
run() { local log="kutuk/fs8_$1.log"; shift; python -u "$@" 2>&1 | grep -v -i "$F" > "$log"; local rc=${PIPESTATUS[0]}; echo "cikis: $rc" >> "$log"; return $rc; }
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
yol() { grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" "$1" | head -1 | tr '\' '/'; }
AD=D_kqgpPe_adsb_42o
BAD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bF8
echo "başladı $(date +%T)"
python - <<'PY' || dur "yerleştirme"
import pandas as pd
P = "data_final/preds/"
SEKIZ = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM", "EHAM", "LSZH"]
d = pd.read_parquet(P + "TEST_B_tum_dpdcb6c5.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
for ad, kay in (("tum6M_adsb", "tum6M_adsb"), ("tum6M_adsb_lirfte", "tum6M_adsb_lirfte_mu2_tk")):
    k = pd.read_parquet(P + f"TEST_B_{kay}.parquet")
    if "ADEP_mvt" not in k:
        k = k.merge(pd.read_parquet(P + "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt")
    k = k.merge(d, on="MVT_ID_mvt", how="left")
    m = k.ADEP_mvt.isin(SEKIZ).to_numpy()
    assert k.loc[m, "yeni"].notna().all()
    for c in ("taban", "taksi"):
        if c in k:
            k[c] = k[c].astype(float)
            k.loc[m, c] = k.loc[m, "yeni"]
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_bF8.parquet", index=False)
    print(ad, "→ _bF8:", int(m.sum()), "satır yeni tabanla")
PY
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 run b5 scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 \
  kaynak=tum6M_adsb_bF8 guncel=tum6M_adsb_lirfte_bF8 ad=$BAD || dur "B5"
echo "B5 tamam $(date +%T)"
for v in "_ha_g2lgbcb_g2t v334 v320_yapısı" "_ha_g2 - G2_yalnız_LightGBM" "_ha_g2lgbcb - G2b_yok" "_ha - G2_yok" "@ - G2_ve_H1_yok"; do
  set -- $v; e=${1/@/}; k=$2; a=$3
  python scripts/g_gonderim.py ${AD}${e} 1.0 b=$BAD lobt=1 "FINAL (4 ay) 8 havalimanlı B (EHAM / LSZH yeni tabanda); A = ${AD}${e} (${a//_/ }). Ara dosya (Y1'siz)." > kutuk/fs8_g_${e:-duz}.log 2>&1 || dur "gönderim $e"
  ara=$(yol kutuk/fs8_g_${e:-duz}.log); [ -n "$ara" ] || dur "ara dosya yok $e"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. FINAL, 8 havalimanlı B, A ${a//_/ }." > kutuk/fs8_k_${e:-duz}.log 2>&1 || dur "Y1 $e"
  son=$(yol kutuk/fs8_k_${e:-duz}.log)
  echo "A${e:-_duz} (${a//_/ }): ara $ara → $son $(grep -o 'MD5: `[0-9a-f]*`' kutuk/fs8_k_${e:-duz}.log | head -1)"
done
python - <<'PY'
import numpy as np, pandas as pd
a = pd.read_parquet("gonderim_final/strong-moose_final_a2.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC"])
import glob, os
son = sorted(glob.glob("gonderim_final/strong-moose_final_a*.parquet"), key=os.path.getmtime)[-1]
b = pd.read_parquet(son)
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_8")).merge(r, on="MVT_ID_mvt")
x["f"] = x.TAXITIME_SEC_mvt_8 - x.TAXITIME_SEC_mvt
print("son dosya", son, "− final_a2 (B satırları, havalimanı × ay):")
print(x[x.NM_TAXI_DIFF_SEC.isna() & (x.f.abs() > 1e-3)].groupby(["ADEP_mvt", "MVT_MONTH"]).f.agg(["size", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
PY
echo "zincir bitti $(date +%T)"

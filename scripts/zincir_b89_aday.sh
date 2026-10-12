#!/bin/bash
# 09.10 öğle (09.10 LB): v316 düzeninde Model B'nin yeni tabanına EHAM / LSZH (8 havalimanı, _L8) ve
# ayrıca EDDF (9, _L9) — v316'nın B'si EHAM / LSZH / EDDF'te v126 B + B5. A tarafı, LIRF (yeni μ, 24 sa yok), Y1 1σ v316 ile aynı.
# + A düzelticisi parkb'siz (_40o) adayı (v330 = _42o ile çift → parkb'nin LB payı).
# Kayıt: kutuk/zincir_b89_aday.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_Y1_ESIK=1.0 PRC_LIRFTE=tum6M_adsb_lirfte_mu2 PRC_A_AD=D_kqgpPe_adsb_42o_ha_g2lgbcb_g2t
echo "başladı $(date +%T)"
python scripts/g_gonderim.py D_kqgpPe_adsb_40o 1.0 b=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5L lobt=1 "A tarafı D_kqgpPe_adsb_40o (düzeltici parkb'siz; H1 / G2 yok) + v316'nın B'si. v330 (_42o) ile farkı yalnız parkb. Ara dosya." > kutuk/aa_40o_gonderim.log 2>&1
ARA=$(grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" kutuk/aa_40o_gonderim.log | head -1 | tr '\' '/')
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 1σ havalimanı + pist + LOBT. v330 ile farkı yalnız A düzelticisi _42o → _40o (parkb yok)." > kutuk/aa_40o_kalibre.log 2>&1
echo "_40o: $ARA → $(grep -o 'gonderim[\/]strong-moose_v[0-9]*\.parquet' kutuk/aa_40o_kalibre.log | head -1) $(grep -o 'MD5: `[0-9a-f]*`' kutuk/aa_40o_kalibre.log | head -1)"
for v in "L8 EDDM,EGLL,LEBL,LEMD,LFPG,LTFM,EHAM,LSZH" "L9 EDDM,EGLL,LEBL,LEMD,LFPG,LTFM,EHAM,LSZH,EDDF"; do
  set -- $v
  cp data/preds/TEST_B_tum_dpdcb6c5L.parquet data/preds/TEST_B_tum_dpdcb6c5$1.parquet
  PRC_B_APT=$2 bash scripts/zincir_b_durum2_uretim.sh 1 _dpdcb6c5$1 > kutuk/zincir_b_dpdcb6c5$1_uretim.log 2>&1
  echo "$1: $(grep 'aday:' kutuk/zincir_b_dpdcb6c5$1_uretim.log)"
done
python - <<'PY'
import glob, re, numpy as np, pandas as pd
a = pd.read_parquet("gonderim/strong-moose_v316.parquet")
r = pd.read_parquet("data/base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"])
for ek in ("L8", "L9"):
    s = open(f"kutuk/zincir_b_dpdcb6c5{ek}_uretim.log", encoding="utf-8").read()
    m = re.findall(r"→ (gonderim[\/]strong-moose_v\d+\.parquet)", s)
    if not m: print(ek, "aday yok"); continue
    b = pd.read_parquet(m[-1].replace("\\", "/"))
    x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_y")).merge(r, on="MVT_ID_mvt")
    x["f"] = x.TAXITIME_SEC_mvt_y - x.TAXITIME_SEC_mvt
    g = x[x.f.abs() > 1e-3].groupby("ADEP_mvt").f.agg(["size", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1)
    print(ek, m[-1], "− v316:"); print(g.to_string())
PY
echo "zincir bitti $(date +%T)"

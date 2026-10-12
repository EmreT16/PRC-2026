#!/bin/bash
# 10.10: CV (4 çeyrek, 8 havalimanı, 24 sa hariç) 651,7 → 646,3 (w = 0,6;
# 4/4 çeyrek, 7/8 havalimanı; kutuk/b_mu_nn_degerlendir.log). μ = 0,4 · melez (0,5 LightGBM + 0,25 + 0,25 CatBoost) + 0,6 · TabM
# (2 tohum × 12 epok), tavan 5.000; LIRF düzeni değişmez (LIRF μ ayrı: tum6M_adsb_lirfte_mu2). A tarafı = v345 / a48'inki
# (D_kqgpPeT2_adsb_42o). Adaylar: normal kipte v345 + B TabM; final kipte a48 + B TabM (8 havalimanlı B).
# Kayıt: kutuk/zincir_b_nn_aday.log (+ kutuk/bn_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0
W=0.6
NN="durum=park,ktah,kom,gs,pc,yte epok=12 tohum=42,7"
NNF=TEST_B_mu_nn_dparkktahkomgspcyte_e12_t2.parquet
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
yerlestir() {  # $1 = preds klasörü, $2 = kaynak melez, $3 = çıktı eki
  python - "$1" "$2" "$3" "$W" "$NNF" <<'PY' || return 1
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from src import model_b as MB
P, kay, ek, w, nnf = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
d = pd.read_parquet(P + f"TEST_B_{kay}.parquet")
n = d[["MVT_ID_mvt"]].merge(pd.read_parquet(P + nnf), on="MVT_ID_mvt", how="left", validate="one_to_one")["bilesen_mu_nn"].to_numpy(float)
assert np.isfinite(n).all(), "TabM μ eksik"
mu0 = d["bilesen_mu"].to_numpy(float)
d["bilesen_mu_melez"] = mu0; d["bilesen_mu_nn"] = n
d["bilesen_mu"] = (1 - w) * mu0 + w * n
K = ["mu", "p24", "yt_tum", "d", "d24"]
eski = d["taksi"].to_numpy(float)
d["taksi"] = MB.combine_tum({k: d[f"bilesen_{k}"].to_numpy(float) for k in K}, d["ADEP_mvt"], yt="yt_tum", cap_non_lirf=5000.0)
d.to_parquet(P + f"TEST_B_tum_{ek}.parquet", index=False)
f = d["taksi"].to_numpy(float) - eski
print(f"TEST_B_tum_{ek}: {len(d):,} satır | μ korelasyonu {np.corrcoef(mu0, n)[0, 1]:.4f} | taksi farkı RMS {np.sqrt(np.mean(f ** 2)):.1f}, ort {f.mean():+.1f}")
print(pd.DataFrame({"apt": d.ADEP_mvt, "f": f}).groupby("apt").f.agg(["size", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
PY
  return ${PIPESTATUS[0]}
}
echo "başladı $(date +%T)"
# ---------------------------------------------------------------- 1. test TabM μ (normal, final)
test -f data/preds/$NNF || python -u scripts/b_mu_nn.py uretim $NN > kutuk/bn_test_normal.log 2>&1 || dur "TabM μ normal"
FP=""
test -f data_final/preds/$NNF || { PRC_FINAL=1 python -u scripts/b_mu_nn.py uretim $NN > kutuk/bn_test_final.log 2>&1 & FP=$!; }
# ---------------------------------------------------------------- 2. normal kip: v345 + B TabM
yerlestir data/preds/ tum_dpdcb6c5L dpdcb6c5N8 || dur "melez normal"
PRC_LIRFTE=tum6M_adsb_lirfte_mu2 PRC_A_AD=D_kqgpPeT2_adsb_42o PRC_B_APT=EDDM,EGLL,LEBL,LEMD,LFPG,LTFM,EHAM,LSZH \
  bash scripts/zincir_b_durum2_uretim.sh 1 _dpdcb6c5N8 > kutuk/bn_normal.log 2>&1
grep -q "zincir bitti" kutuk/bn_normal.log || dur "normal B zinciri"
SON=$(grep -o "aday: .*" kutuk/bn_normal.log); echo "normal $SON"
python - "$(echo $SON | grep -o 'gonderim[\/]strong-moose_v[0-9]*\.parquet' | tail -1 | tr '\\' '/')" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1])
for k in ("v345", "v343"):
    a = pd.read_parquet(f"gonderim/strong-moose_{k}.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
    d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
    print(f"  {sys.argv[1]} − {k}: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, maks |fark| {np.abs(d).max():.0f}")
PY
# ---------------------------------------------------------------- 3. final kip: a48 + B TabM
[ -z "$FP" ] || wait $FP || dur "TabM μ final"
test -f data_final/preds/$NNF || dur "TabM μ final yok"
export PRC_FINAL=1
yerlestir data_final/preds/ tum_dpdcb6c5 dpdcb6c5N || dur "melez final"
python - <<'PY' || dur "yerleştirme final"
import pandas as pd
P = "data_final/preds/"
SEKIZ = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM", "EHAM", "LSZH"]
d = pd.read_parquet(P + "TEST_B_tum_dpdcb6c5N.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
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
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_bN8.parquet", index=False)
    print(ad, "→ _bN8:", int(m.sum()), "satır TabM'li melezle")
PY
BAD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN8
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 \
  kaynak=tum6M_adsb_bN8 guncel=tum6M_adsb_lirfte_bN8 ad=$BAD 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/bn_b5_final.log
test -f data_final/preds/TEST_B_$BAD.parquet || dur "B5 final"
python scripts/g_gonderim.py D_kqgpPeT2_adsb_42o 1.0 b=$BAD lobt=1 "FINAL (4 ay): A = D_kqgpPeT2_adsb_42o (TabM, ikinci aşamasız) + 8 havalimanlı B, B μ'ye TabM (w 0,6). a48 ile tek fark B μ. Ara dosya (Y1'siz)." > kutuk/bn_g_final.log 2>&1 || dur "gönderim final"
ARA=$(grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" kutuk/bn_g_final.log | head -1 | tr '\\' '/')
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist (1σ) + LOBT. FINAL: a48 + B μ'ye TabM." > kutuk/bn_k_final.log 2>&1 || dur "Y1 final"
SONF=$(grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" kutuk/bn_k_final.log | head -1 | tr '\\' '/')
echo "final: ara $ARA → $SONF $(grep -o 'MD5: `[0-9a-f]*`' kutuk/bn_k_final.log | head -1)"
python - "$SONF" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim_final/strong-moose_final_a48.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC"])
assert len(b) == 670_790 and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_n")).merge(r, on="MVT_ID_mvt")
x["f"] = x.TAXITIME_SEC_mvt_n - x.TAXITIME_SEC_mvt
B = x.NM_TAXI_DIFF_SEC.isna()
print(f"  − a48: B satırları {int(B.sum()):,} RMS {np.sqrt((x.f[B] ** 2).mean()):.1f}, ort {x.f[B].mean():+.1f}; A satırları RMS {np.sqrt((x.f[~B] ** 2).mean()):.2f}")
print(x[B].groupby(["MVT_MONTH"]).f.agg(["size", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
PY
echo "zincir bitti $(date +%T)"

#!/bin/bash
# 10.10 gece: LB'de B μ'ye TabM CV'nin ~4 katı taşındı (v347 −0,858). Test B satırlarının bir
# kısmı eğitim dağılımı dışında (EHAM Ocak aksaması vb.); orada tohum oynaklığı büyük (08.10: μ LightGBM tek ↔ 3 tohum RMS 71 sn)
# → B μ TabM'i 2 → 5 tohum (2026, 1, 2 eklenir; ağırlık 2/5 + 3/5) ile varyans azaltımı (v316'daki μ 3 tohum mantığı; CV ölçemez).
# Adaylar (tek fark B μ TabM tohum sayısı): normal v355'in (A _42pT), final a56'nın karşılığı.
# final_tabm.sh denemesi bitmeden başlamaz (ekran kartı / bellek). Kayıt: kutuk/zincir_b_mu_t5.log (+ kutuk/bt5_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
until grep -q "zincir bitti\|HATA" kutuk/final_tabm.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
NN="durum=park,ktah,kom,gs,pc,yte epok=12 tohum=2026,1,2"
K=TEST_B_mu_nn_dparkktahkomgspcyte_e12
test -f data/preds/${K}_t3.parquet || python -u scripts/b_mu_nn.py uretim $NN > kutuk/bt5_test_normal.log 2>&1 || dur "TabM μ normal"
test -f data_final/preds/${K}_t3.parquet || PRC_FINAL=1 python -u scripts/b_mu_nn.py uretim $NN > kutuk/bt5_test_final.log 2>&1 || dur "TabM μ final"
python - "$K" <<'PY' || dur "ortalama"
import sys, numpy as np, pandas as pd
K = sys.argv[1]
for P in ("data/preds/", "data_final/preds/"):
    a = pd.read_parquet(P + f"{K}_t2.parquet"); b = pd.read_parquet(P + f"{K}_t3.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    n2, n3 = a.bilesen_mu_nn.to_numpy(float), b.bilesen_mu_nn.to_numpy(float)
    a["bilesen_mu_nn"] = 0.4 * n2 + 0.6 * n3
    a.to_parquet(P + f"{K}_t5.parquet", index=False)
    print(f"{P}{K}_t5: {len(a):,} satır | 2 ↔ 3 tohum korelasyonu {np.corrcoef(n2, n3)[0, 1]:.4f}, fark RMS {np.sqrt(np.mean((n2 - n3) ** 2)):.1f}")
PY
echo "TabM μ 5 tohum tamam $(date +%T)"
# ---------------------------------------------------------------- normal: v355 + B μ TabM 5 tohum
python scripts/b_mu_nn_yerlestir.py data/preds tum_dpdcb6c5L tum_dpdcb6c5N85 nn=${K}_t5.parquet > kutuk/bt5_yerlestir_normal.log 2>&1 || dur "yerleştirme normal"
PRC_LIRFTE=tum6M_adsb_lirfte_mu2 PRC_A_AD=D_kqgpPeT2_adsb_42pT PRC_B_APT=EDDM,EGLL,LEBL,LEMD,LFPG,LTFM,EHAM,LSZH \
  bash scripts/zincir_b_durum2_uretim.sh 1 _dpdcb6c5N85 > kutuk/bt5_normal.log 2>&1
grep -q "zincir bitti" kutuk/bt5_normal.log || dur "normal B zinciri"
SON=$(grep -o "aday: .*" kutuk/bt5_normal.log); echo "normal $SON"
python - "$(echo $SON | grep -o 'strong-moose_v[0-9]*\.parquet' | tail -1)" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet("gonderim/" + sys.argv[1]); a = pd.read_parquet("gonderim/strong-moose_v355.parquet")
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all() and len(b) == 344_841 and b.TAXITIME_SEC_mvt.notna().all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"  {sys.argv[1]} − v355: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, maks |fark| {np.abs(d).max():.0f}")
PY
# ---------------------------------------------------------------- final: a56 + B μ TabM 5 tohum
export PRC_FINAL=1
python scripts/b_mu_nn_yerlestir.py data_final/preds tum_dpdcb6c5 tum_dpdcb6c5N5 nn=${K}_t5.parquet > kutuk/bt5_yerlestir_final.log 2>&1 || dur "yerleştirme final"
python - <<'PY' || dur "8 havalimanlı yerleştirme final"
import pandas as pd
P = "data_final/preds/"
SEKIZ = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM", "EHAM", "LSZH"]
d = pd.read_parquet(P + "TEST_B_tum_dpdcb6c5N5.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
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
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_bN85.parquet", index=False)
    print(ad, "→ _bN85:", int(m.sum()), "satır")
PY
BAD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN85
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 python -u scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 \
  kaynak=tum6M_adsb_bN85 guncel=tum6M_adsb_lirfte_bN85 ad=$BAD 2>&1 | grep -v -i "$F" > kutuk/bt5_b5_final.log
test -f data_final/preds/TEST_B_$BAD.parquet || dur "B5 final"
python scripts/g_gonderim.py D_kqgpPeT2_adsb_42pT 1.0 b=$BAD lobt=1 "FINAL (4 ay): a56 + B μ TabM 5 tohum (2 + 3; varyans azaltımı). Ara dosya (Y1'siz)." > kutuk/bt5_g_final.log 2>&1 || dur "gönderim final"
ARA=gonderim_final/$(grep -o "strong-moose_final_a[0-9]*\.parquet" kutuk/bt5_g_final.log | head -1)
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist (1σ) + LOBT. FINAL: a56 + B μ TabM 5 tohum." > kutuk/bt5_k_final.log 2>&1 || dur "Y1 final"
SONF=gonderim_final/$(grep -o "strong-moose_final_a[0-9]*\.parquet" kutuk/bt5_k_final.log | grep -v "$(basename $ARA)" | head -1)
echo "final: ara $ARA → $SONF $(grep -o 'MD5: `[0-9a-f]*`' kutuk/bt5_k_final.log | head -1)"
python - "$SONF" <<'PY'
import sys, numpy as np, pandas as pd
b = pd.read_parquet(sys.argv[1]); a = pd.read_parquet("gonderim_final/strong-moose_final_a56.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC"])
assert len(b) == 670_790 and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_n")).merge(r, on="MVT_ID_mvt")
x["f"] = x.TAXITIME_SEC_mvt_n - x.TAXITIME_SEC_mvt
B = x.NM_TAXI_DIFF_SEC.isna()
print(f"  − a56: B satırları RMS {np.sqrt((x.f[B] ** 2).mean()):.1f}, ort {x.f[B].mean():+.2f}; A satırları RMS {np.sqrt((x.f[~B] ** 2).mean()):.2f}")
PY
echo "zincir bitti $(date +%T)"

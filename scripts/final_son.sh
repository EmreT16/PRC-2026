#!/bin/bash
# Final aşaması (08.10): Model A nesillerinden (scripts/final_a_nesil.sh) sonra v320 yapısının
# geri kalanı, PRC_FINAL=1 (test = final_ranking, data_final/, adaylar gonderim_final/).
# v320 = A: kqgpPe harmanı → ADS-B düzeltici _42o (parkb, LIRF yankı hedge'i β 0,25) → H1 → G2 + G2b (LightGBM + CatBoost)
#        B: melez (μ = 0,5 · LightGBM + 0,25 · CatBoost 1. çalıştırma + 0,25 · CatBoost 2. çalıştırma [hepsi], 3 tohum;
#           p24 / yt / d park + ktah modelinden; LIRF dışı tavan 5.000) 6 havalimanında; EHAM / LSZH / EDDF tabanı v126 B
#           (scripts/final_b_taban.py); LIRF: b_lirf_te (anahtar hy,park; μ = 0,5 · LightGBM + 0,5 · CatBoost 2. çalıştırma),
#           ön koşullu satırlar v126 B; 24 sa tarih kayması karışımı (b_lirf_tarih_kaymasi.py); B5 (8 havalimanı, k 1,25)
#        G: gönderim + LOBT penceresi → Y1 havalimanı + pist (eşik 1σ)
# Komutlar v316 / v320 üretimindekilerle aynı (zincir_parkb_uretim.sh, kuyruk_cpu_08.sh, zincir_b_durum2_uretim.sh).
# Kayıt: kutuk/final_son.log (+ adım başına kutuk/fs_<adım>.log)
cd "$(dirname "$0")/.."  # 09.10: göreli kök (depo / tekrar kopyası)
export PYTHONIOENCODING=utf-8 PRC_FINAL=1
P=data_final/preds
F='warn\|X\["tip"\]'
run() { local log="kutuk/fs_$1.log"; shift; python -u "$@" 2>&1 | grep -v -i "$F" > "$log"; local rc=${PIPESTATUS[0]}; echo "cikis: $rc" >> "$log"; return $rc; }
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
adim() { echo "[$(date +%T)] $*"; }

# ---------------------------------------------------------------- 0. ADS-B gün özellikleri (2026: Oca, Şub, Haz, Tem)
# ADS-B noktaları önceden indirilmiş olmalı (scripts/adsb_hat.py); gün sayısı aşağıda denetlenir.
adim "ADS-B gün özellikleri"
PRC_ADSB_GEVSEK=1 PRC_ADSB_IZ5=1 PRC_ADSB_OZ=adsb5 run adsb_oz scripts/adsb_ozellik.py hepsi isci=6 || dur "ADS-B özellikleri"
python - <<'PY' || dur "ADS-B gün sayısı"
from pathlib import Path
g = sorted(p.stem[3:] for p in Path("data_final/adsb5").glob("oz_2026-*.parquet"))
ay = {m: sum(x[5:7] == m for x in g) for m in ("01", "02", "06", "07")}
print("2026 ADS-B gün özelliği:", ay, "| fazladan:", [x for x in g if x[5:7] not in ay])
assert ay == {"01": 31, "02": 28, "06": 30, "07": 31}, ay
PY

# ---------------------------------------------------------------- 1. Model A nesilleri + park dizisi
until grep -qE "zincir bitti|HATA" kutuk/final_a_nesil.log 2>/dev/null; do sleep 120; done
grep -q HATA kutuk/final_a_nesil.log && dur "Model A nesilleri"
adim "park dizisi (park_b_oz)"
run park_b scripts/park_b_ozellik.py || dur "park_b_oz"

# ---------------------------------------------------------------- 2. A (işlemci) ve B (ekran kartı) kolları paralel
AD=D_kqgpPe_adsb_42o
(
  export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
  D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
  PRC_EPOCH_TAM=1 run a_duz scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_42o \
  && PRC_DUZ_CV=parkb run a_h1 scripts/havalimani_agirlik.py $AD \
  && PRC_DUZ_CV=parkb PRC_G2_MODEL=lgb+cb run a_g2 scripts/gecikme_asama2.py ${AD}_ha ha=1 \
  && PRC_DUZ_CV=parkb PRC_G2_MODEL=lgb+cb PRC_G2_TERS=1 run a_g2t scripts/gecikme_asama2.py ${AD}_ha ha=1
  echo "A: $? $(date +%T)" > kutuk/fs_a.done
) &
(
  # 09.10 akşam: PRC_FS_B_ATLA=1 → tüm satırlı B üretimleri (ADS-B kullanmıyor) atlanır, mevcut TEST_B_tum_* dosyaları
  # kullanılır (ADS-B onarımından sonra yalnız ADS-B'ye bağlı adımların yeniden koşusu)
  if [ "${PRC_FS_B_ATLA:-0}" = "1" ]; then
    for a in tum_dparkktah tum_dpdcb3ham tum_dpdcb3ham_hepsi; do test -f $P/TEST_B_$a.parquet || exit 1; done
    echo "B: 0 (atlandı, mevcut dosyalar) $(date +%T)" > kutuk/fs_b.done; exit 0
  fi
  run b_pk scripts/b_uretim_tum.py durum=park,ktah ad=tum_dparkktah \
  && run b_mu1 scripts/b_uretim_tum.py durum=park,ktah,kom,gs,pc,yte tohum=3 mu_cb=1 cb_tohum=3 ad=tum_dpdcb3ham \
  && run b_mu2 scripts/b_uretim_tum.py hepsi durum=park,ktah,kom,gs,pc,yte tohum=3 mu_cb=1 cb_tohum=3 ad=tum_dpdcb3ham_hepsi
  echo "B: $? $(date +%T)" > kutuk/fs_b.done
) &
wait
grep -q "A: 0" kutuk/fs_a.done || dur "A kolu"
grep -q "B: 0" kutuk/fs_b.done || dur "B kolu"
adim "A: G2 + G2b birleştirme"
python - "$AD" <<'PY' || dur "G2 + G2b"
import sys, numpy as np, pandas as pd
P, ad = "data_final/preds/", sys.argv[1]
h, g, t = (pd.read_parquet(P + f"TEST_{ad}_ha{e}.parquet") for e in ("", "_g2lgbcb", "_g2tlgbcb"))
assert (h.MVT_ID_mvt.to_numpy() == g.MVT_ID_mvt.to_numpy()).all() and (h.MVT_ID_mvt.to_numpy() == t.MVT_ID_mvt.to_numpy()).all()
r1, r2 = g.taksi.to_numpy(float) - h.taksi.to_numpy(float), t.taksi.to_numpy(float) - h.taksi.to_numpy(float)
assert not ((np.abs(r1) > 1e-9) & (np.abs(r2) > 1e-9)).any(), "G2 ve G2b kümeleri ayrık değil"
o = g.copy(); o["taksi"] = np.maximum(60.0, h.taksi.to_numpy(float) + r1 + r2)
o.to_parquet(P + f"TEST_{ad}_ha_g2lgbcb_g2t.parquet", index=False)
print(f"G2 {(np.abs(r1) > 1e-9).sum():,} satır (RMS {np.sqrt(np.mean(r1 ** 2)):.1f}), G2b {(np.abs(r2) > 1e-9).sum():,} satır (RMS {np.sqrt(np.mean(r2 ** 2)):.1f})")
PY

# ---------------------------------------------------------------- 3. B: melez, LIRF, 24 sa karışımı, eski tabana yerleştirme
adim "B: melez (tavan 5.000)"
python - <<'PY' || dur "melez"
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from src import model_b as MB
P = "data_final/preds/"
pk, h1, h2 = (pd.read_parquet(P + f"TEST_B_{a}.parquet") for a in ("tum_dparkktah", "tum_dpdcb3ham", "tum_dpdcb3ham_hepsi"))
assert (pk.MVT_ID_mvt.to_numpy() == h1.MVT_ID_mvt.to_numpy()).all()
h2 = pk[["MVT_ID_mvt"]].merge(h2, on="MVT_ID_mvt", how="left", validate="one_to_one")
print("LightGBM μ iki çalıştırmada aynı mı:", float(np.abs(h1.bilesen_mu - h2.bilesen_mu).max()),
      "| CatBoost μ iki çalıştırma RMS farkı:", round(float(np.sqrt(np.mean((h1.bilesen_mu_cb - h2.bilesen_mu_cb) ** 2))), 1))
o = pk.copy()
o["bilesen_mu"] = 0.5 * h1.bilesen_mu.to_numpy() + 0.25 * h1.bilesen_mu_cb.to_numpy() + 0.25 * h2.bilesen_mu_cb.to_numpy()
K = ["mu", "p24", "yt_tum", "d", "d24"]
o["taksi"] = MB.combine_tum({k: o[f"bilesen_{k}"].to_numpy(float) for k in K}, o["ADEP_mvt"], yt="yt_tum", cap_non_lirf=5000.0)
o.to_parquet(P + "TEST_B_tum_dpdcb6c5.parquet", index=False)
print("melez:", len(o), "satır | ort", round(o.taksi.mean(), 1), "| 5.000 tavanında", int((o.taksi >= 5000).sum()))
PY
adim "B: LIRF (yeni μ) + 24 sa tarih kayması"
PRC_LIRF_MU=tum_dpdcb3ham_hepsi PRC_LIRF_EK=_mu2 run b_lirf scripts/b_lirf_te.py uretim || dur "b_lirf_te"
run b_tk scripts/b_lirf_tarih_kaymasi.py uretim tum6M_adsb_lirfte_mu2 _tk || dur "tarih kayması"
python - <<'PY' || dur "eski tabana yerleştirme"
import pandas as pd
P = "data_final/preds/"
ALTI = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
d = pd.read_parquet(P + "TEST_B_tum_dpdcb6c5.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
for ad, kay in (("tum6M_adsb", "tum6M_adsb"), ("tum6M_adsb_lirfte", "tum6M_adsb_lirfte_mu2_tk")):
    k = pd.read_parquet(P + f"TEST_B_{kay}.parquet")
    if "ADEP_mvt" not in k:
        k = k.merge(pd.read_parquet(P + "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt")
    k = k.merge(d, on="MVT_ID_mvt", how="left")
    m = k.ADEP_mvt.isin(ALTI).to_numpy()
    assert k.loc[m, "yeni"].notna().all()
    for c in ("taban", "taksi"):
        if c in k:
            k[c] = k[c].astype(float)
            k.loc[m, c] = k.loc[m, "yeni"]
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_bF.parquet", index=False)
    print(ad, "→ _bF:", int(m.sum()), "satır yeni tabanla")
PY
BAD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bF
adim "B5 → $BAD"
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_HARMAN=kqgpxg PRC_DUZ_EK=split_4,split_5,split_6 PRC_EPOCH_TAM=1 run b5 scripts/adsb_b_duzeltici5.py uretim \
  apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1.25 iz5=1 nmsiz=1 damga5=1 opdi=1 opdi_satir=1 \
  kaynak=tum6M_adsb_bF guncel=tum6M_adsb_lirfte_bF ad=$BAD || dur "B5"

# ---------------------------------------------------------------- 4. gönderim + Y1 (1σ) + LOBT
adim "gönderim"
python scripts/g_gonderim.py ${AD}_ha_g2lgbcb_g2t 1.0 b=$BAD lobt=1 "FINAL (4 ay): v320 yapısı, scripts/final_son.sh. Ara dosya (Y1'siz)." > kutuk/fs_g1.log 2>&1 || dur "gönderim"
ARA=$(grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" kutuk/fs_g1.log | head -1 | tr '\\' '/')
[ -n "$ARA" ] || dur "ara dosya yok"
PRC_Y1_ESIK=1.0 python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist (1σ) + LOBT. FINAL adayı (v320 yapısı)." > kutuk/fs_g2.log 2>&1 || dur "Y1"
SON=$(grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" kutuk/fs_g2.log | head -1 | tr '\\' '/')
adim "aday: ara $ARA → $SON $(grep -o 'MD5: `[0-9a-f]*`' kutuk/fs_g2.log | head -1)"

# ---------------------------------------------------------------- 5. denetim: Oca / Tem satırlarında final adayı ↔ v320 (normal test)
python - "$SON" <<'PY'
import os, sys, numpy as np, pandas as pd
if not os.path.exists("gonderim/strong-moose_v320.parquet"):
    sys.exit(0)  # karşılaştırma yalnız normal test gönderimi (v320) varken; yoksa atlanır
f = pd.read_parquet(sys.argv[1]); e = pd.read_parquet("gonderim/strong-moose_v320.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC"])
x = f.merge(r, on="MVT_ID_mvt")
print("satır", len(f), "| boş", int(f.TAXITIME_SEC_mvt.isna().sum()), "| en küçük", float(f.TAXITIME_SEC_mvt.min()))
print("ay başına ortalama:", x.groupby("MVT_MONTH").TAXITIME_SEC_mvt.mean().round(1).to_dict())
y = x.merge(e, on="MVT_ID_mvt", suffixes=("", "_v320"))
y["f"] = y.TAXITIME_SEC_mvt - y.TAXITIME_SEC_mvt_v320
y["B"] = y.NM_TAXI_DIFF_SEC.isna()
print(f"Oca/Tem {len(y):,} satır: final − v320 RMS {np.sqrt((y.f ** 2).mean()):.1f}, ort {y.f.mean():+.2f}")
print(y.groupby(["B", "ADEP_mvt"]).f.agg(["size", lambda s: float(np.sqrt((s ** 2).mean())), "mean"]).round(1).to_string())
PY
adim "bitti"

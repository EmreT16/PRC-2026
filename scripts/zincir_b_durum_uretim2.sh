#!/bin/bash
# 05.10 akşam, zincir_b_durum_uretim.sh'ın düzeltilmiş devamı (ilkinde float32 ataması hatası; v229 GEÇERSİZ).
# B5 düzeltici bugünkü kodla eski dosyayı birebir vermiyor (RMS 10–26 sn) → fark yöntemi: yeni B = v224'teki B
# + (B5_bugün(yeni taban) − B5_bugün(eski taban)); böylece yalnız Model B tabanının değişimi taşınır. Her adım denetimli.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
set -e
echo "başladı $(date +%T)"
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
ALTI = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
d = pd.read_parquet(P + "TEST_B_tum_dkomgspcyte.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
adep = pd.read_parquet(P + "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"])
for ad in ("tum6M_adsb", "tum6M_adsb_lirfte"):
    k = pd.read_parquet(P + f"TEST_B_{ad}.parquet")
    k = (k if "ADEP_mvt" in k else k.merge(adep, on="MVT_ID_mvt")).merge(d, on="MVT_ID_mvt", how="left")
    m = k.ADEP_mvt.isin(ALTI).to_numpy()
    assert k.loc[m, "yeni"].notna().all()
    for c in ("taban", "taksi"):
        if c in k:
            k[c] = k[c].astype("float64")
            k.loc[m, c] = k.loc[m, "yeni"].astype("float64")
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_bdurum.parquet", index=False)
    print(ad, "→ _bdurum:", int(m.sum()), "satır değişti")
PY
export PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
[ -f data/preds/TEST_B_D_kqgpxg_17a_b5_bdurum_ham.parquet ] || python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 kaynak=tum6M_adsb_bdurum guncel=tum6M_adsb_lirfte_bdurum ad=D_kqgpxg_17a_b5_bdurum_ham 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/b5_bdurum.log
unset PRC_HARMAN PRC_ADSB_OZ PRC_ADSB_GEVSEK
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
eski = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5.parquet")
r0 = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_yeniden.parquet", columns=["MVT_ID_mvt", "taksi"]).rename(columns={"taksi": "r0"})
r1 = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_bdurum_ham.parquet", columns=["MVT_ID_mvt", "taksi"]).rename(columns={"taksi": "r1"})
x = eski.merge(r0, on="MVT_ID_mvt", validate="one_to_one").merge(r1, on="MVT_ID_mvt", validate="one_to_one")
x["eski_son"] = x["taksi"]
ALTI = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
# fark yalnız 6 havalimanına: B5'in satırlar arası özellikleri yüzünden diğer havalimanlarında da 7–11 sn RMS oynama
# oluyor (taban değişmediği hâlde); temiz tek değişken için onlar v224'teki gibi kalır.
m6 = x.ADEP_mvt.isin(ALTI).to_numpy()
x["taksi"] = np.where(m6, np.maximum(60.0, x["taksi"] + (x["r1"] - x["r0"])), x["taksi"])
f = x["taksi"] - x["eski_son"]
assert (f[~m6].abs() < 1e-6).all(), "6 havalimanı dışında değişiklik var"
assert x["taksi"].notna().all()
x.drop(columns=["r0", "r1", "eski_son"]).to_parquet(P + "TEST_B_D_kqgpxg_17a_b5_bdurum.parquet", index=False)
print(f"yeni B: {int((f.abs() > 1e-6).sum())} satır değişti, ort {f.mean():+.1f}, RMS {np.sqrt((f**2).mean()):.1f}")
print(x.assign(f=f).groupby("ADEP_mvt").f.agg(["size", lambda s: int((s.abs() > 1e-6).sum()), "mean", lambda s: float(np.sqrt((s**2).mean()))]).round(1).to_string())
PY
python scripts/g_gonderim.py D_kqgpPe_adsb_37a 1.0 b=D_kqgpxg_17a_b5_bdurum lobt=1 "v223 + Model B durum özellikleri (tüm satırlı modele komşu aralıkları, günlük seriler, pist konfigürasyonu, yankı geçmişi; yalnız EDDM, EGLL, LEBL, LEMD, LFPG, LTFM NM'siz satırları; B5 farkı aynı kodla). Ara dosya; kalibre edilmiş hali gönderilir." > kutuk/b_durum_gonderim.log 2>&1
grep -E "Dosya|MD5|model_b" kutuk/b_durum_gonderim.log
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1)
grep -q "$(basename $SON)" kutuk/b_durum_gonderim.log || { echo "gönderim dosyası bulunamadı; durdu"; exit 1; }
python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist katmanı + LOBT. v224 ile farkı = yalnız Model B durum özellikleri (6 havalimanının NM'siz satırları)." 2>&1 | tail -n 3
echo "zincir bitti $(date +%T)"

#!/bin/bash
# 05.10 akşam: Model B tüm satırlı modele durum özellikleri (kom, gs, pc, yte) — üretim.
# 1) taban yeniden üretimi (eski TEST_B_tum ile aynı mı: tekrarlanabilirlik), 2) durumlu tahmin, 3) 6 havalimanının tabanı
# yenisiyle değiştirilir (EHAM / LSZH / EDDF v126, LIRF v167 aynen), 4) B5 ADS-B düzeltici (önce eski kaynakla: eski dosyayı
# birebir veriyor mu; sonra yeni kaynakla), 5) gönderim = v223 tabanı + yeni B (lobt=1) → Y1 havalimanı + pist.
# Kayıt: kutuk/zincir_b_durum_uretim.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
python -u scripts/b_uretim_tum.py ad=tum_yeniden > kutuk/b_uretim_taban_yeniden.log 2>&1 || { echo "taban üretimi hatalı"; exit 1; }
python -u scripts/b_uretim_tum.py durum=kom,gs,pc,yte ad=tum_dkomgspcyte > kutuk/b_uretim_durum.log 2>&1 || { echo "durum üretimi hatalı"; exit 1; }
tail -n 14 kutuk/b_uretim_durum.log
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
ALTI = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"]
eski, yen = pd.read_parquet(P + "TEST_B_tum.parquet"), pd.read_parquet(P + "TEST_B_tum_yeniden.parquet")
f = np.abs(eski.taksi.to_numpy() - yen.taksi.to_numpy())
print(f"taban yeniden üretim: maks fark {f.max():.4f} sn ({'BİREBİR' if f.max() < 1e-6 else 'FARKLI'})")
d = pd.read_parquet(P + "TEST_B_tum_dkomgspcyte.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
for ad in ("tum6M_adsb", "tum6M_adsb_lirfte"):
    k = pd.read_parquet(P + f"TEST_B_{ad}.parquet")
    if "ADEP_mvt" not in k:
        k = k.merge(pd.read_parquet(P + "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt")
    k = k.merge(d, on="MVT_ID_mvt", how="left")
    m = k.ADEP_mvt.isin(ALTI).to_numpy()
    assert k.loc[m, "yeni"].notna().all()
    for c in ("taban", "taksi"):
        if c in k:
            k.loc[m, c] = k.loc[m, "yeni"]
    k.drop(columns=["yeni"]).to_parquet(P + f"TEST_B_{ad}_bdurum.parquet", index=False)
    print(ad, "→ _bdurum:", int(m.sum()), "satır değişti")
PY
export PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 ad=D_kqgpxg_17a_b5_yeniden 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/b5_yeniden.log
python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 kaynak=tum6M_adsb_bdurum guncel=tum6M_adsb_lirfte_bdurum ad=D_kqgpxg_17a_b5_bdurum 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/b5_bdurum.log
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5.parquet"); b = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_yeniden.parquet")
c = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_bdurum.parquet")
x = a.merge(b[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", suffixes=("", "_y")).merge(c[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "taksi_d"}), on="MVT_ID_mvt")
print(f"B5 yeniden üretim: maks fark {np.abs(x.taksi - x.taksi_y).max():.4f} sn")
x["f"] = x.taksi_d - x.taksi
print(x.groupby("ADEP_mvt").f.agg(["size", lambda s: int((s.abs() > 1e-6).sum()), "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
PY
unset PRC_HARMAN PRC_ADSB_OZ PRC_ADSB_GEVSEK
python scripts/g_gonderim.py D_kqgpPe_adsb_37a 1.0 b=D_kqgpxg_17a_b5_bdurum lobt=1 "v223 + Model B durum özellikleri (tüm satırlı modele komşu aralıkları, günlük seriler, pist konfigürasyonu, yankı geçmişi; EDDM, EGLL, LEBL, LEMD, LFPG, LTFM; B5 düzeltici yeniden). Ara dosya; kalibre edilmiş hali gönderilir." 2>&1 | grep -E "Dosya|MD5|model_b"
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1)
python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist katmanı + LOBT. v224 ile farkı = yalnız Model B durum özellikleri (6 havalimanının NM'siz satırları)." 2>&1 | tail -n 3
echo "zincir bitti $(date +%T)"

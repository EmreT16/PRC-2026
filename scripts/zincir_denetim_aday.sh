#!/bin/bash
# 05.10 gece: CV'si iyi çıkan denetim maddelerinden iki aday, ikisi de v224'e karşı tek değişkenli.
#  A) P1: Model A düzelticisine durum özellikleri (durum=1; CV 198,03 → 197,59). Harman kqgpPe (eski hedge p'si; tek fark durum).
#  B) T2 + T3: B5 bugünkü kodla ve iz5=1 ile yeniden (tekrar üretilebilir B5; CV 8 havalimanının 7'sinde iyi), taban eski B.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
set -e
echo "başladı $(date +%T)"
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
export PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe
python -u scripts/adsb_duzeltici.py uretim $D durum=1 k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_39a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpPe_39a.log
unset PRC_HARMAN PRC_DUZ_EK
python scripts/g_gonderim.py D_kqgpPe_adsb_39a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "v223 + Model A düzelticisine durum özellikleri (komşu aralıkları, komşuların tahmini taksisi, günlük seriler, pist konfigürasyonu, yankı geçmişi; durum=1; CV 198,03 → 197,59). Ara dosya." > kutuk/da_a_gonderim.log 2>&1
grep -E "Dosya|MD5" kutuk/da_a_gonderim.log
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1); grep -q "$(basename $SON)" kutuk/da_a_gonderim.log
python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist + LOBT. v224 ile farkı = yalnız Model A düzelticisinde durum özellikleri." 2>&1 | tail -n 3
export PRC_HARMAN=kqgpxg
python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 iz5=1 ad=D_kqgpxg_17a_b5_iz5 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/b5_iz5_uretim.log
unset PRC_HARMAN PRC_ADSB_OZ PRC_ADSB_GEVSEK
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5.parquet"); b = pd.read_parquet(P + "TEST_B_D_kqgpxg_17a_b5_iz5.parquet")
x = a.merge(b[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", suffixes=("", "_y")); x["f"] = x.taksi_y - x.taksi
print("B5 bugünkü kod + iz5 − 02.10 dosyası:")
print(x.groupby("ADEP_mvt").f.agg(["size", lambda s: int((s.abs() > 1e-6).sum()), "mean", lambda s: float(np.sqrt((s**2).mean()))]).round(1).to_string())
PY
python scripts/g_gonderim.py D_kqgpPe_adsb_37a 1.0 b=D_kqgpxg_17a_b5_iz5 lobt=1 "v223 + B5 düzeltici bugünkü kodla ve iz5=1 ile yeniden (tekrar üretilebilir B5; CV 8 havalimanının 7'sinde iyi). Ara dosya." > kutuk/da_b_gonderim.log 2>&1
grep -E "Dosya|MD5" kutuk/da_b_gonderim.log
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1); grep -q "$(basename $SON)" kutuk/da_b_gonderim.log
python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist + LOBT. v224 ile farkı = yalnız B5 (bugünkü kod + iz5)." 2>&1 | tail -n 3
echo "zincir bitti $(date +%T)"

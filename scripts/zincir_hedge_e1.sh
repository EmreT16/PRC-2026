#!/bin/bash
# 05.10 gece (denetim): LIRF yankı hedge'i final E1'in p_kilit'iyle. v224 ile tek fark bu (izli LIRF satırları).
# Düzeltici aynı ayarlarla yeniden (ek=_38a); önce _37a ile birebir aynı mı denetlenir (yalnız hedge satırları farklı olmalı).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_HARMAN=kqgpPe1 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
set -e
echo "başladı $(date +%T)"
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_38a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpPe_38a.log
grep "yankı hedge" kutuk/adsb_uretim_D_kqgpPe_38a.log kutuk/adsb_uretim_D_kqgpPe_37a.log
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_D_kqgpPe_adsb_37a.parquet"); b = pd.read_parquet(P + "TEST_D_kqgpPe_adsb_38a.parquet")
r = pd.read_parquet("data/base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
x = a[["MVT_ID_mvt", "taksi"]].merge(b[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", suffixes=("_a", "_b")).merge(r, on="MVT_ID_mvt")
x["f"] = x.taksi_b - x.taksi_a; d = x[x.f.abs() > 1e-6]
print(f"değişen {len(d):,} satır; havalimanı dışı LIRF: {int((d.ADEP_mvt != 'LIRF').sum())}")
print(d.groupby(["ADEP_mvt", "MVT_MONTH"]).f.agg(["size", "mean", lambda s: float(np.sqrt((s**2).mean()))]).round(2).to_string())
assert (d.ADEP_mvt == "LIRF").all(), "LIRF dışında fark var (düzeltici tekrar üretimi birebir değil)"
PY
unset PRC_HARMAN
python scripts/g_gonderim.py D_kqgpPe_adsb_38a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "v223 + LIRF yankı hedge'inin p_kilit'i final E1'den (pistli, 6 tohum). Ara dosya." > kutuk/hedge_e1_gonderim.log 2>&1
grep -E "Dosya|MD5" kutuk/hedge_e1_gonderim.log
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1)
grep -q "$(basename $SON)" kutuk/hedge_e1_gonderim.log || { echo "gönderim dosyası bulunamadı; durdu"; exit 1; }
python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist + LOBT. v224 ile farkı = yalnız LIRF yankı hedge'inin p_kilit kaynağı (eski kqgp E1 → final E1)." 2>&1 | tail -n 3
echo "zincir bitti $(date +%T)"

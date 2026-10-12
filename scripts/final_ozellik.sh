#!/bin/bash
# Final aşaması (08.10): Model A / B'nin modelden bağımsız özellik tabloları, PRC_FINAL=1 ile
# (test = final_ranking.parquet, çıktılar data_final/). Önkoşul: scripts/final_taban.py (dep_train / dep_rank / V3),
# scripts/final_hazirlik.py (eğitim tarafı kopyaları), taban tablolar (features_v126.base_table). Modele bağlı özellikler
# (komsu_tah_oz ← 1. nesil, komsu2_oz ← 2. nesil, park_b_oz ← final harmanı) ve ADS-B gün özellikleri ayrı adımlarda.
# Sıra bağımlılıkları: gecikme_oz → gecikme_pist_oz; wx_oz → varis_wx_oz; taban → yanki_te_oz.
# Kayıt: kutuk/final_ozellik.log (+ adım başına kutuk/final_oz_<betik>.log)
cd "$(dirname "$0")/.."  # 09.10: göreli kök (depo / tekrar kopyası)
export PYTHONIOENCODING=utf-8 PRC_FINAL=1
echo "başladı $(date +%T)"
for b in kuyruk_ozellik gecikme_ozellik gecikme_pist_ozellik wx_ozellik varis_wx_ozellik pencere_ileri_ozellik \
         park_varis_ozellik komsu_ozellik yanki_te_ozellik gunluk_seri_ozellik pist_ozellik opdi_ozellik; do
  t0=$(date +%s)
  python -u scripts/$b.py > kutuk/final_oz_$b.log 2>&1 || { echo "HATA: $b ($(tail -3 kutuk/final_oz_$b.log | tr '\n' ' '))"; exit 1; }
  echo "$b tamam ($(( $(date +%s) - t0 )) sn)"
done
python - <<'PY'
import pandas as pd, numpy as np
from pathlib import Path
D = Path("data_final")
rk = set(pd.read_parquet(D / "dep_rank.parquet", columns=["MVT_ID_mvt"]).MVT_ID_mvt)
tr = set(pd.read_parquet(D / "dep_train.parquet", columns=["MVT_ID_mvt"]).MVT_ID_mvt)
for f in ("kuyruk_oz", "gecikme_oz", "gecikme_pist_oz", "wx_oz", "varis_wx_oz", "pencere_ileri_oz", "park_varis_oz",
          "komsu_oz", "yanki_te_oz", "gunluk_oz", "pist_oz", "opdi_oz"):
    x = pd.read_parquet(D / f"{f}.parquet")
    t, e = x[x.MVT_ID_mvt.isin(rk)], x[x.MVT_ID_mvt.isin(tr)]
    sut = [c for c in x.columns if c != "MVT_ID_mvt"]
    print(f"{f:<17} test {len(t):>7,}/{len(rk):,} | eğitim {len(e):>9,} | boş oranı test {t[sut].isna().mean().mean():.3f}"
          f" eğitim {e[sut].isna().mean().mean():.3f} | sütun {len(sut)}")
PY
echo "bitti $(date +%T)"

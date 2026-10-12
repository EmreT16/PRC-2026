#!/bin/bash
# 10.10: TabM LB'de v341 − v330 = −0,365.
#   1) Epok: split_1, tohum 42, 12 epok (tabm_kqgp_e12) ↔ mevcut 8 epok (tabm_kqgp). Önceden tanımlı kural: düzeltici sonrası
#      Oca / Tem'de (w ızgarasının en iyisi) 12 epok ≥ 0,05 iyiyse E = 12, değilse E = 8.
#   2) İkinci tohum (7), E epokla 6 bölme. E = 12 ise tohum 42 de 12 epokla 2–6 bölmelerde yeniden.
#   3) İki tohumun ortalaması → NN_tabm_kqgp2_<bölme> → 12 ay değerlendirme (scripts/tabm_degerlendir.py tabm_kqgp2 …).
# Üretim dosyalarına dokunmaz (yeni adlar). Kayıt: kutuk/tabm_guclendir.log (+ kutuk/tg_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
T="model=tabm tabm_k=8 amp=1"
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
nn() { local ad=$1; shift; python -u scripts/nn_bilesen.py $ad "$@" $T > kutuk/tg_$ad.log 2>&1 || dur "$ad"; grep "^\[split" kutuk/tg_$ad.log; }
BOL="split_1 split_2 split_3 split_4 split_5 split_6"
echo "başladı $(date +%T)"
nn tabm_kqgp_e12 split_1 epok=12 tohum=42
python -u scripts/tabm_degerlendir.py tabm_kqgp_e12 split_1 > kutuk/tg_deg_e12.log 2>&1 || dur "değerlendirme e12"
E=$(python - <<'PY'
import re
def enk(f):  # düzeltici sonrası tablonun "toplam" sütununda en küçük değer (w ızgarası)
    s = open(f, encoding="utf-8").read().split("--- düzeltici sonrası")[1].split("çapraz")[0]
    return min(float(l.split()[1]) for l in s.splitlines() if re.match(r"^\s*0\.\d+\s", l) and not l.strip().startswith("0.00"))
e8, e12 = enk("kutuk/tabm_pilot_degerlendir.log"), enk("kutuk/tg_deg_e12.log")
print(12 if e12 <= e8 - 0.05 else 8)
import sys; print(f"epok karşılaştırması (split_1, düzeltici sonrası en iyi w): 8 epok {e8:.3f}, 12 epok {e12:.3f}", file=sys.stderr)
PY
)
echo "seçilen epok: $E $(date +%T)"
if [ "$E" = "12" ]; then
  for s in split_2 split_3 split_4 split_5 split_6; do nn tabm_kqgp_e12 $s epok=12 tohum=42; done
  A1=tabm_kqgp_e12
else
  A1=tabm_kqgp
fi
for s in $BOL; do nn tabm_kqgp_t7e$E $s epok=$E tohum=7; done
python - "$A1" "tabm_kqgp_t7e$E" <<'PY' || dur "ortalama"
import sys, pandas as pd
sys.path.insert(0, ".")
from src import config
P = config.DATA_DIR / "preds"
for k in range(1, 7):
    a = pd.read_parquet(P / f"NN_{sys.argv[1]}_split_{k}.parquet"); b = pd.read_parquet(P / f"NN_{sys.argv[2]}_split_{k}.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    a["nn"] = 0.5 * (a["nn"].to_numpy() + b["nn"].to_numpy())
    a.to_parquet(P / f"NN_tabm_kqgp2_split_{k}.parquet", index=False)
print("NN_tabm_kqgp2_split_1…6 yazıldı:", sys.argv[1], "+", sys.argv[2])
PY
python -u scripts/tabm_degerlendir.py tabm_kqgp2 $BOL > kutuk/tg_deg_2tohum.log 2>&1 || dur "değerlendirme 2 tohum"
sed -n '/bölme birlikte/,$p' kutuk/tg_deg_2tohum.log
echo "karşılık (tek tohum, 8 epok):"; sed -n '/bölme birlikte/,$p' kutuk/tabm_oof_degerlendir.log
echo "zincir bitti $(date +%T)"

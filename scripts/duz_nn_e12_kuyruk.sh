#!/bin/bash
# 10.10: 8 epokta eğitim kaybı hâlâ düşüyordu (0,0247 → 0,0239) → 12 epok
# CV'si. Önceden tanımlı kural (TabM tabanındakiyle aynı): v = 0,7 harmanında Oca / Tem ağırlıklı ≥ 0,05 iyi → üretimde 12 epok.
# B5 TabM CV'si (b5_nn_cv.sh) bitmeden başlamaz (bellek / ekran kartı). Kayıt: kutuk/duz_nn_e12_kuyruk.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "^bitti\|HATA" kutuk/b5_nn_cv.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
bash scripts/duz_nn_cv.sh 12
python scripts/duz_nn_degerlendir.py nn=ADSB_duz_cv_parkb_tabm2_nn_e12 2>&1 | grep -v -i warn > kutuk/duz_nn_degerlendir_e12.log
cat kutuk/duz_nn_degerlendir_e12.log
python - <<'PY'
import re
def v07(f):
    for s in open(f, encoding="utf-8"):
        if s.startswith("0.7 "):
            return float(s.split()[3])
a, b = v07("kutuk/duz_nn_degerlendir.log"), v07("kutuk/duz_nn_degerlendir_e12.log")
print(f"v = 0,7 ağırlıklı: 8 epok {a:.3f} → 12 epok {b:.3f} ({b - a:+.3f}) → {'12 EPOK' if b - a <= -0.05 else '8 epok kalır'}")
PY
echo "bitti $(date +%T)"

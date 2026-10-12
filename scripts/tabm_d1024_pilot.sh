#!/bin/bash
# 10.10: Model A tabanındaki TabM'in kapasitesi — d_block 512 → 1024 (k = 8, fp16, 12 epok, tohum 42),
# split_1 pilotu; karşılık NN_tabm_kqgp_e12 (aynı tohum / epok, d 512). Kural scripts/tabm_pilot_kiyas.py (≥ 0,05 ve iki ay iyi)
# → tam koşu (6 bölme × 2 tohum + test). planli üretim zinciri bitmeden başlamaz (bellek / ekran kartı).
# Kayıt: kutuk/tabm_d1024_pilot.log (+ kutuk/td_split1.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0
until grep -q "zincir bitti\|HATA" kutuk/zincir_planli_aday.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 python -u scripts/nn_bilesen.py tabm_kqgp_d1024 split_1 model=tabm tabm_k=8 tabm_d=1024 amp=1 epok=12 tohum=42 \
  > kutuk/td_split1.log 2>&1 || { echo "HATA: TabM d1024 $(date +%T)"; tail -5 kutuk/td_split1.log; exit 1; }
grep "^\[split\|epok 1/\|epok 12/" kutuk/td_split1.log
python scripts/tabm_pilot_kiyas.py tabm_kqgp_e12 tabm_kqgp_d1024 2>&1 | grep -v -i warn
echo "bitti $(date +%T)"

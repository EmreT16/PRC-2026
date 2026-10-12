#!/bin/bash
# 02.10 gece: split_5 (Nisan+Ekim) ve split_6 (Mayıs+Eylül) Model A bileşenleri (kqgpxg'nin
# beşi) + B1. Amaç: ADS-B düzelticisini 12 aya genişletmek (PRC_DUZ_EK=split_4,split_5,split_6). kuyruk_39'un eşi.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run b1_split5 scripts/b1_taban.py split_5 && run b1_split6 scripts/b1_taban.py split_6 || { echo "b1 hatalı $(date +%T)"; exit 1; }
echo "b1 bitti $(date +%T)"
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
(
  run kqgp_lgb_s56 scripts/aday_dogrula.py v_lgb_huber1000_3t_kqgp split_5 split_6 \
  && run kqgp_e1_s56 scripts/aday_dogrula.py v_e1_huber1000_3t_kqgp split_5 split_6
  echo "islemci: $? $(date +%T)" > kutuk/s56_cpu.done
) &
(
  run kqgp_dw_s56 scripts/gcb_dw_tarama.py dw_lr03_ek5_l2_10_kqgp split_5 split_6 \
  && run kqgp_gcb12k_s56 scripts/gcb_varyant.py it12000_kqgp 12000 0.06 9 split_5 split_6 \
  && run kqgp_gcb12k_t2026_s56 scripts/gcb_varyant.py it12000_t2026_kqgp 12000 0.06 9 split_5 split_6 tohum=2026 \
  && run kqgp_gcb12k_ort2_s56 scripts/gcb_ortalama.py it12000_ort2_kqgp it12000_kqgp it12000_t2026_kqgp \
  && run kqgp_xgb1500_s56 scripts/aday_dogrula.py v_xgb_kqgp_1500 split_5 split_6
  echo "ekran: $? $(date +%T)" > kutuk/s56_gpu.done
) &
wait
cat kutuk/s56_cpu.done kutuk/s56_gpu.done
echo "kuyruk bitti $(date +%T)"

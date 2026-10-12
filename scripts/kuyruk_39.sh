#!/bin/bash
# 02.10 akşam: split_4 (Kasım+Mart dışarıda) Model A bileşenleri, kqgpxg harmanının
# beşi. Amaç: ADS-B düzelticisinin eğitimine Kasım+Mart'ı eklemek (PRC_DUZ_S4=1). İşlemci ve ekran kartı paralel.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run b1_split4 scripts/b1_taban.py split_4 || { echo "b1 hatalı $(date +%T)"; exit 1; }
echo "b1 bitti $(date +%T)"
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
(
  run kqgp_lgb_s4 scripts/aday_dogrula.py v_lgb_huber1000_3t_kqgp split_4 \
  && run kqgp_e1_s4 scripts/aday_dogrula.py v_e1_huber1000_3t_kqgp split_4
  echo "islemci: $? $(date +%T)" > kutuk/s4_cpu.done
) &
(
  run kqgp_dw_s4 scripts/gcb_dw_tarama.py dw_lr03_ek5_l2_10_kqgp split_4 \
  && run kqgp_gcb12k_s4 scripts/gcb_varyant.py it12000_kqgp 12000 0.06 9 split_4 \
  && run kqgp_gcb12k_t2026_s4 scripts/gcb_varyant.py it12000_t2026_kqgp 12000 0.06 9 split_4 tohum=2026 \
  && run kqgp_gcb12k_ort2_s4 scripts/gcb_ortalama.py it12000_ort2_kqgp it12000_kqgp it12000_t2026_kqgp \
  && run kqgp_xgb1500_s4 scripts/aday_dogrula.py v_xgb_kqgp_1500 split_4
  echo "ekran: $? $(date +%T)" > kutuk/s4_gpu.done
) &
wait
cat kutuk/s4_cpu.done kutuk/s4_gpu.done
echo "kuyruk bitti $(date +%T)"

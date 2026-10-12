#!/bin/bash
# split_3 (Aralık+Haziran 2025, diğer 10 ay eğitim): v149 Model A harmanının (D_dw) beş bileşeninin dışarıda bırakılmış
# tahminleri. Amaç: ADS-B düzelticisinin eğitimini Ocak/Şubat/Temmuz/Ağustos'a ek olarak Aralık/Haziran ile genişletmek
# (2 → 4 ay CV'de −0,85 verdi). Bileşen tarifleri kuyruk_10/12/17 ile aynı; ağırlıklar değişmez (D_dw).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run b1_split3 scripts/b1_taban.py split_3 \
 && run gcb_it6000_split3 scripts/gcb_varyant.py it6000 6000 0.06 9 split_3 \
 && run gcb_it6000_t2026_split3 scripts/gcb_varyant.py it6000_t2026 6000 0.06 9 split_3 tohum=2026 \
 && run gcb_ortalama_it6000_ort2_split3 scripts/gcb_ortalama.py it6000_ort2 it6000 it6000_t2026 \
 && run aday_v_e1_huber1000_3t_split3 scripts/aday_dogrula.py v_e1_huber1000_3t split_3 \
 && run aday_v_lgb_huber1000_3t_split3 scripts/aday_dogrula.py v_lgb_huber1000_3t split_3 \
 && run gcb_dw_split3 scripts/gcb_dw_tarama.py dw_lr03_ek5_l2_10 split_3

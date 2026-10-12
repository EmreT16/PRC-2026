#!/bin/bash
# 12 aylık global CB (PID 20108) bitince: E1 (12 ay) → beş bileşenli test harmanı (kuyruk_5 gönderimi üretir)
# → global CB'nin uzun eğitim denemesi (6.000 iterasyon). Süreçler tek tek; GPU ve RAM paylaşılmaz.
cd "$(dirname "$0")/.."
while tasklist.exe //FI "PID eq 20108" //NH 2>/dev/null | grep -q 20108; do sleep 30; done
[ -f data/preds/TEST_gcb.parquet ] || { echo "TEST_gcb yok" > kutuk/kuyruk_7_hata.log; exit 1; }
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run d_tam_e1_lgb42 scripts/d_tam_egitim.py tam_e1_lgb42 && run g_bes_bilesen scripts/g_bes_bilesen.py
sleep 90   # kuyruk_5'in gönderim dosyasını yazması için
run gcb_it6000 scripts/gcb_varyant.py it6000 6000 0.06 9

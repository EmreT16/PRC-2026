#!/bin/bash
# Son tur doğrulamaları (düzenek, iki bölme): E1 Huber 3 tohum; global CB 6.000 it. ikinci tohum (2026).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run aday_v_e1_huber1000_3t scripts/aday_dogrula.py v_e1_huber1000_3t
run gcb_it6000_t2026 scripts/gcb_varyant.py it6000_t2026 6000 0.06 9 tohum=2026

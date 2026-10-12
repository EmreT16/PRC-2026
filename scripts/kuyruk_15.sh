#!/bin/bash
# Global CB ayar elemesi (3.000 it., iki bölme) ↔ A2 global CB (aynı ayarlar, 3.000 it.):
# border_count 254 (GPU varsayılanı 128); Depthwise ağaç (min_data_in_leaf 50). Kuyruk 14 bitince başlar.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "cikis:" kutuk/k_gcb_nan.log 2>/dev/null; do sleep 30; done
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run gcb_border254 scripts/gcb_varyant.py border254 3000 0.06 9 cb:border_count=254
run gcb_depthwise scripts/gcb_varyant.py depthwise 3000 0.06 9 cb:grow_policy=Depthwise cb:min_data_in_leaf=50
run k_gcb_border254 scripts/karsilastir.py A2_global_cb:global_cb GCB_border254:global_cb
run k_gcb_depthwise scripts/karsilastir.py A2_global_cb:global_cb GCB_depthwise:global_cb

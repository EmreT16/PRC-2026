#!/bin/bash
# v150: Model A = v149 (dondurulan, D_dw); Model B = LIRF'te v126, LIRF dışında tüm satırlarla eğitilen model (b_uretim_tum.py).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run b_uretim_tum scripts/b_uretim_tum.py \
 && run g_gonderim_D_dw_Btum scripts/g_gonderim.py D_dw 1.0 b=tum "Model A: v149 ile aynı (dondurulan). Model B: LIRF v126; LIRF dışı tüm satırlarla eğitilen model (μ ve yer tutucu tüm satırlardan, 24 saat B satırlarından, 3500 tavan)"

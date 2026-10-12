#!/bin/bash
# Model B: tüm satırlı modele depo temel tablosunun 11 fiziksel ek sütunu (stand–pist mesafesi, kalkış sayıları, pist
# bilgileri). Kuyruk 20 bitince başlar.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "cikis:" kutuk/b_kayma_cv_varyant.log 2>/dev/null; do sleep 30; done
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run b_model_cv_tum_ek scripts/b_model_cv_tum.py ek=1
run b_kayma_cv_ek scripts/b_kayma_cv.py BM_cv_oof:v126_B BM_cv_tum_oof:tum_mu_yt_tavan3500 BM_cv_tum_ek_oof:tum_mu_yt_tavan3500

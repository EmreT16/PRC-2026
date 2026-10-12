#!/bin/bash
# Model B: tüm satırlı modelin varyantları (4 çeyrekli CV): B satırı ağırlığı 10 / 30, μ için Huber δ=600, ikisi birlikte.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
run b_model_cv_tum_w10 scripts/b_model_cv_tum.py agirlik=10
run b_model_cv_tum_w30 scripts/b_model_cv_tum.py agirlik=30
run b_model_cv_tum_h600 scripts/b_model_cv_tum.py huber=600
run b_model_cv_tum_w10_h600 scripts/b_model_cv_tum.py agirlik=10 huber=600
run b_kayma_cv_varyant scripts/b_kayma_cv.py BM_cv_oof:v126_B BM_cv_tum_oof:tum_mu_yt_tavan3500 \
  BM_cv_tum_w10_oof:tum_mu_yt_tavan3500 BM_cv_tum_w30_oof:tum_mu_yt_tavan3500 BM_cv_tum_h600_oof:tum_mu_yt_tavan3500 \
  BM_cv_tum_w10_h600_oof:tum_mu_yt_tavan3500

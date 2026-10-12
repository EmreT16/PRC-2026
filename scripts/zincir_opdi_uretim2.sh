#!/bin/bash
# 07.10: ikinci OPDI adayı (_39o bitince): opdi=1 + opdi_satir=1 + PRC_EPOCH_TAM=1 (CV: 196,64 → 196,58; epoch 198,03 → 197,92).
cd "$(dirname "$0")/.."
until grep -qE "zincir bitti|durdu|Error|error" kutuk/zincir_opdi_uretim_39o.log 2>/dev/null; do sleep 20; done
bash scripts/zincir_opdi_uretim.sh _40o 1 "OPDI özellikleri + OPDI'nin yerde gördüğü izsiz satırlar + mvt_epoch 12 ay (Y8)" opdi=1 opdi_satir=1

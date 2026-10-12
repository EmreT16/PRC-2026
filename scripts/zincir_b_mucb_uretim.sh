#!/bin/bash
# 08.10: Model B μ = 0,5 · LightGBM μ + 0,5 · CatBoost μ (durum park,ktah,kom,gs,pc,yte); p24 / yt / d park + ktah modelinden
# (v295'teki melez düzen). CV (6 havalimanı, 24 sa hariç; kutuk/zincir_b_mucb_cv.log): v295 769,8 → 756,8; 4/4 çeyrek,
# 6/6 havalimanı. Üretim: b_uretim_tum.py mu_cb=1 → TEST_B_tum_dpdcbham (LightGBM μ'si TEST_B_tum_dpd ile aynı olmalı:
# yeniden üretim denetimi) → TEST_B_tum_dpdcb (melez) → zincir_b_durum2_uretim.sh 1 _dpdcb. Kayıt: kutuk/zincir_b_mucb_uretim.log
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
echo "başladı $(date +%T)"
test -f data/preds/TEST_B_tum_dpdcbham.parquet || python -u scripts/b_uretim_tum.py durum=park,ktah,kom,gs,pc,yte mu_cb=1 ad=tum_dpdcbham 2>&1 | grep -v -i warn > kutuk/b_uretim_dpdcbham.log
test -f data/preds/TEST_B_tum_dpdcbham.parquet || { echo "B üretimi hatalı"; tail -5 kutuk/b_uretim_dpdcbham.log; exit 1; }
echo "B üretimi tamam $(date +%T)"
python - <<'PY'
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from src import model_b as MB
P = "data/preds/"
a = pd.read_parquet(P + "TEST_B_tum_dparkktah.parquet"); h = pd.read_parquet(P + "TEST_B_tum_dpdcbham.parquet")
d = pd.read_parquet(P + "TEST_B_tum_dpd.parquet")
assert np.array_equal(a.MVT_ID_mvt.to_numpy(), h.MVT_ID_mvt.to_numpy()) and np.array_equal(a.MVT_ID_mvt.to_numpy(), d.MVT_ID_mvt.to_numpy())
print(f"denetim: LightGBM μ (yeni) − TEST_B_tum_dpd μ: maks |fark| {np.abs(h.bilesen_mu - d.bilesen_mu).max():.4f} sn")
print(f"μ korelasyonu lgb / cb: {np.corrcoef(h.bilesen_mu, h.bilesen_mu_cb)[0, 1]:.4f}; ort lgb {h.bilesen_mu.mean():.1f}, cb {h.bilesen_mu_cb.mean():.1f}")
o = a.copy()
o["bilesen_mu"] = 0.5 * h["bilesen_mu"].to_numpy() + 0.5 * h["bilesen_mu_cb"].to_numpy()
K = ["mu", "p24", "yt_b", "yt_tum", "d", "d24"]
o["taksi"] = MB.combine_tum({k: o[f"bilesen_{k}"].to_numpy(float) for k in K}, o["ADEP_mvt"], yt="yt_tum", cap_non_lirf=3500.0)
o.to_parquet(P + "TEST_B_tum_dpdcb.parquet", index=False)
m = pd.read_parquet(P + "TEST_B_tum_dpdmu.parquet")
f = o.taksi - m.taksi
print("v295'in B'sine (dpdmu) göre fark:")
print(pd.DataFrame({"apt": o.ADEP_mvt, "f": f}).groupby("apt").f.agg(["size", "mean", lambda s: float(np.sqrt((s ** 2).mean()))]).round(1).to_string())
PY
bash scripts/zincir_b_durum2_uretim.sh 1 _dpdcb
python - <<'PY'
import numpy as np, pandas as pd
import glob, os
son = sorted(glob.glob("gonderim/strong-moose_v*.parquet"), key=lambda f: int(f.split("_v")[1].split(".")[0]))[-1]
a = pd.read_parquet("gonderim/strong-moose_v295.parquet"); b = pd.read_parquet(son)
assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
d = b.TAXITIME_SEC_mvt.to_numpy(float) - a.TAXITIME_SEC_mvt.to_numpy(float)
print(f"{os.path.basename(son)} − v295: değişen {(np.abs(d) > 1e-3).sum():,} satır, RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, |fark| > 300 sn {int((np.abs(d) > 300).sum())}, |fark| > 1 sn {int((np.abs(d) > 1).sum())}")
PY
echo "zincir bitti $(date +%T)"

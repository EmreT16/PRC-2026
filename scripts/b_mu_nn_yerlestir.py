"""10.10: B tahmin dosyasında μ'yü TabM ile harmanlar ve karışımı yeniden kurar.
μ = (1 − w) · melez (0,5 LightGBM + 0,25 + 0,25 CatBoost) + w · TabM; taksi = combine_tum (24 sa karışımı, tavan 5.000; LIRF hariç
tavan). LIRF düzeni değişmez (LIRF ayrı dosyadan gelir). scripts/zincir_b_nn_aday.sh'deki yerleştirmenin bağımsız hâli.
Kullanım: python scripts/b_mu_nn_yerlestir.py <preds klasörü> <kaynak ek> <çıktı ek> [w=0.6] [nn=TEST_B_mu_nn_…parquet]
  ör. python scripts/b_mu_nn_yerlestir.py data_final/preds tum_dpdcb6c5 tum_dpdcb6c5N
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model_b as MB  # noqa: E402

NNF = "TEST_B_mu_nn_dparkktahkomgspcyte_e12_t2.parquet"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    poz = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    P, kay, ek = Path(poz[0]), poz[1], poz[2]
    w = float(opt.get("w", 0.6))
    d = pd.read_parquet(P / f"TEST_B_{kay}.parquet")
    n = d[["MVT_ID_mvt"]].merge(pd.read_parquet(P / opt.get("nn", NNF)), on="MVT_ID_mvt", how="left",
                                validate="one_to_one")["bilesen_mu_nn"].to_numpy(float)
    assert np.isfinite(n).all(), "TabM μ eksik"
    mu0 = d["bilesen_mu"].to_numpy(float)
    d["bilesen_mu_melez"] = mu0
    d["bilesen_mu_nn"] = n
    d["bilesen_mu"] = (1 - w) * mu0 + w * n
    eski = d["taksi"].to_numpy(float)
    d["taksi"] = MB.combine_tum({k: d[f"bilesen_{k}"].to_numpy(float) for k in ("mu", "p24", "yt_tum", "d", "d24")},
                                d["ADEP_mvt"], yt="yt_tum", cap_non_lirf=5000.0)
    d.to_parquet(P / f"TEST_B_{ek}.parquet", index=False)
    f = d["taksi"].to_numpy(float) - eski
    print(f"TEST_B_{ek}: {len(d):,} satır | w = {w} | μ korelasyonu {np.corrcoef(mu0, n)[0, 1]:.4f} | "
          f"taksi farkı RMS {np.sqrt(np.mean(f ** 2)):.1f}, ort {f.mean():+.1f}")


if __name__ == "__main__":
    main()

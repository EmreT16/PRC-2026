"""11.10: scripts/b_lirf_mu_esik.py'nin kabul ettiği kural (10.10 gece;
iç içe çeyrek seçimi −1,1, 3/4 çeyrek; t = 0,95, w = 0,6) üretimde: LIRF NM'siz test satırlarında, ön koşul dışı ve p0 > 0,95
ise μ = 0,4 · μ_yeni + 0,6 · μ_TabM. μ_yeni = 0,5 · bilesen_mu + 0,5 · bilesen_mu_cb (TEST_B_tum_dpdcb3ham_hepsi; üretimdeki
PRC_LIRF_MU), μ_TabM = TEST_B_mu_nn_dparkktahkomgspcyte_e12_t5_hepsi (5 tohum). Karışım doğrusal → taksi_yeni = max(60, taksi +
p0 · w · (μ_TabM − μ_yeni)) (bu satırlarda alt sınır etkin değil; denetlenir).
Girdi TEST_B_lirf_te_mu2 + TEST_B_tum6M_adsb_lirfte_mu2 → çıktı TEST_B_lirf_te_mu2n + TEST_B_tum6M_adsb_lirfte_mu2n.
Kullanım: python scripts/b_lirf_mu_nn_uygula.py   (PRC_FINAL=1 → data_final)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

P = config.DATA_DIR / "preds"
ESIK, W = 0.95, 0.6


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    L = pd.read_parquet(P / "TEST_B_lirf_te_mu2.parquet")
    T = pd.read_parquet(P / "TEST_B_tum6M_adsb_lirfte_mu2.parquet")
    M = pd.read_parquet(P / "TEST_B_tum_dpdcb3ham_hepsi.parquet")[["MVT_ID_mvt", "bilesen_mu", "bilesen_mu_cb"]]
    N = pd.read_parquet(P / "TEST_B_mu_nn_dparkktahkomgspcyte_e12_t5_hepsi.parquet")[["MVT_ID_mvt", "bilesen_mu_nn"]]
    x = L.merge(M, on="MVT_ID_mvt", how="left", validate="one_to_one").merge(N, on="MVT_ID_mvt", how="left", validate="one_to_one")
    mu = (0.5 * x.bilesen_mu + 0.5 * x.bilesen_mu_cb).to_numpy(float)
    nn = x.bilesen_mu_nn.to_numpy(float)
    p0 = x.p0.to_numpy(float)
    m = (p0 > ESIK) & ~x.on_kosul.to_numpy(bool)
    assert np.isfinite(mu[m]).all() and np.isfinite(nn[m]).all(), "μ eksik"
    taksi = x.taksi.to_numpy(float)
    assert (taksi[m] - p0[m] * mu[m] >= 0).all() and (taksi[m] > 60).all(), "alt sınır etkin satır var"
    d = np.where(m, p0 * W * (nn - mu), 0.0)
    yeni = np.where(m, np.maximum(60.0, taksi + d), taksi)
    print(f"LIRF test B {len(x)} satır | p0 > {ESIK} ve ön koşul dışı: {int(m.sum())} | μ_TabM − μ_yeni ort {np.mean((nn - mu)[m]):+.1f}, "
          f"RMS {np.sqrt(np.mean(((nn - mu)[m]) ** 2)):.1f} | taksi farkı ort {np.mean(d[m]):+.1f}, RMS {np.sqrt(np.mean(d[m] ** 2)):.1f}, "
          f"maks |fark| {np.abs(d).max():.0f}")
    L2 = L.copy()
    L2["taksi"] = yeni
    L2.to_parquet(P / "TEST_B_lirf_te_mu2n.parquet", index=False)
    t = T.merge(L2[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"}), on="MVT_ID_mvt", how="left")
    assert len(t) == len(T) and (t.MVT_ID_mvt.to_numpy() == T.MVT_ID_mvt.to_numpy()).all()
    lirf = t.yeni.notna().to_numpy()
    assert np.allclose(t.taksi.to_numpy()[lirf], L.set_index("MVT_ID_mvt").loc[t.MVT_ID_mvt[lirf], "taksi"].to_numpy())
    t["taksi"] = t.yeni.fillna(t.taksi).astype(float)
    t.drop(columns=["yeni"]).to_parquet(P / "TEST_B_tum6M_adsb_lirfte_mu2n.parquet", index=False)
    f = t.taksi.to_numpy() - T.taksi.to_numpy()
    print(f"TEST_B_tum6M_adsb_lirfte_mu2n: {len(t)} B satırı, değişen {int((np.abs(f) > 1e-6).sum())}")


if __name__ == "__main__":
    main()

"""LIRF 24 saat ön koşulu satırlarında gecikme bandına göre olasılıklı karışım (Model B; eğitim verisinden kural).

Ön koşul (src/model_b.py on_kosul_24s): kalkış tarifeden sonraki güne sarkmış ve uçak tipi boş; eğitimde 68 satır (hepsi
LIRF), 12'si 24 saat kaymalı (hedef ≥ 80.000), 53'ü yer tutucu (hedef ≈ gecikme). Eğitimde gecikme [15,5 sa, 24 sa)
bandında satırların çoğu 24 saat kaymalı, bandın altında neredeyse hiçbiri (29.09 ölçümü: 9/13 ↔ 1/53).
v126 bu satırlara ≈ gecikme (yer tutucu) tahmin eder. Burada bant içindeki test satırlarına MSE'yi en küçükleyen karışım:
  tahmin = p·(24 saat hedef ortalaması) + (1 − p)·gecikme,  p = (n_24 + 1) / (n + 2)  (Laplace düzeltmesi)
Bandın dışındaki satırlar (24 saatin üstü dahil: orada iki senaryo da ≈ gecikme) v126'da kalır.
Çıktı: data/preds/TEST_B_on_kosul_bant.parquet (MVT_ID_mvt, ADEP_mvt, taksi, v126, gecikme, p).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

BANT = (15.5 * 3600, 24 * 3600)
V126 = config.PRC4_DIR / "submissions" / "strong-moose_v126.parquet"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr, rk = MB.load_b("train"), MB.load_b("rank")
    t = tr[MB.on_kosul_24s(tr)]
    y, d = t[MB.TARGET_B].to_numpy(float), t[MB.DELAY].to_numpy(float)
    s24 = y >= 80000
    band = (d >= BANT[0]) & (d < BANT[1])
    n, n24 = int(band.sum()), int((band & s24).sum())
    p = (n24 + 1) / (n + 2)
    m24 = float(y[band & s24].mean())
    print(f"Eğitim ön koşul satırı {len(t)} | bant [{BANT[0] / 3600:.1f} sa, {BANT[1] / 3600:.0f} sa): {n} satır, "
          f"24 saat {n24} → p = {p:.3f}; 24 saat hedef ortalaması {m24:,.0f} | bant altı: "
          f"{int((~band & (d < BANT[0]) & s24).sum())}/{int((d < BANT[0]).sum())} 24 saat")
    r = rk[MB.on_kosul_24s(rk)].reset_index(drop=True)
    rd = r[MB.DELAY].to_numpy(float)
    rb = (rd >= BANT[0]) & (rd < BANT[1])
    out = r.loc[rb, ["MVT_ID_mvt", "ADEP_mvt"]].copy()
    out["gecikme"] = rd[rb]
    out["taksi"] = p * m24 + (1 - p) * out["gecikme"]
    out = out.merge(pd.read_parquet(V126), on="MVT_ID_mvt", how="left", validate="one_to_one").rename(
        columns={"TAXITIME_SEC_mvt": "v126"})
    out["p"] = p
    assert out["taksi"].notna().all() and out["v126"].notna().all()
    out.to_parquet(config.DATA_DIR / "preds" / "TEST_B_on_kosul_bant.parquet", index=False)
    print(f"Test ön koşul satırı {len(r)} | bantta {len(out)}:")
    print(out[["MVT_ID_mvt", "gecikme", "v126", "taksi"]].round(0).to_string(index=False))
    dlt = 87575.0 - out["gecikme"].to_numpy()
    print(f"Beklenen SSE kazancı (eğitim oranları geçerliyse): {float(np.sum(p ** 2 * dlt ** 2)):.3e} "
          f"→ satır başına MSE {float(np.sum(p ** 2 * dlt ** 2)) / 344841:,.0f}")
    V.log_experiment({"ad": "TEST_B_on_kosul_bant", "tur": "Model B", "bant_sa": [BANT[0] / 3600, BANT[1] / 3600],
                      "egitim_bant_n": n, "egitim_bant_24s": n24, "p": p, "hedef_24s_ort": m24, "test_satir": len(out)})


if __name__ == "__main__":
    main()

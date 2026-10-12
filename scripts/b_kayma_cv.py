"""Model B — test–eğitim kaymasına göre ağırlıklı CV değerlendirmesi (eğitim yok, OOF dosyalarından).

Neden: NM'siz satırlarda gecikme (MVT − tarife) dağılımı testte farklı (medyan oranı EHAM 1,51, LSZH 2,43). 4 çeyrekli CV
2025 içinde kaldığı için bunu göremez; LB geçmişi (v67/v88/v97) EHAM/LSZH/EDDF'de CV'ye ters düştü.
Yöntem: her havalimanında eğitim B satırlarına, eğitim gecikmesinin ondalık kutularına (10 kutu) göre
ağırlık = testteki kutu payı / eğitimdeki kutu payı. Ağırlıklı ve ağırlıksız RMSE, 24 saat satırları (hedef ≥ 80.000) hariç.
Kullanım: python scripts/b_kayma_cv.py <dosya:sütun> [<dosya:sütun> ...]
  ör. python scripts/b_kayma_cv.py BM_cv_oof:v126_B BM_cv_tum_oof:tum_mu_yt_tavan3500   (dosya: data/preds/<dosya>.parquet)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB  # noqa: E402

PRED = config.DATA_DIR / "preds"
GROUPS = {"6 havalimanı": ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM"], "EHAM": ["EHAM"], "LSZH": ["LSZH"],
          "EDDF": ["EDDF"], "LIRF (ön koşul dışı)": ["LIRF"]}


def shift_weights(tr, rk, n_bins=10):
    w = np.ones(len(tr))
    for apt in tr["ADEP_mvt"].unique():
        ti = (tr["ADEP_mvt"] == apt).to_numpy()
        a = tr.loc[ti, MB.DELAY].to_numpy(float)
        b = rk.loc[rk["ADEP_mvt"] == apt, MB.DELAY].to_numpy(float)
        if len(b) == 0:
            continue
        edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1))[1:-1])
        ka, kb = np.digitize(a, edges), np.digitize(b, edges)
        pa = np.bincount(ka, minlength=len(edges) + 1) / len(a)
        pb = np.bincount(kb, minlength=len(edges) + 1) / len(b)
        w[ti] = (pb / np.maximum(pa, 1e-9))[ka]
    return w


def main(specs):
    sys.stdout.reconfigure(encoding="utf-8")
    tr = MB.load_b("train")
    rk = MB.load_b("rank")
    tr["on_kosul"] = MB.on_kosul_24s(tr)
    w = shift_weights(tr, rk)
    y = tr[MB.TARGET_B].to_numpy(float)
    keep = (y < 80000) & ~tr["on_kosul"].to_numpy()
    preds = {}
    for s in specs:
        f, col = s.split(":", 1)
        preds[s] = tr[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED / f"{f}.parquet", columns=["MVT_ID_mvt", col]),
                                            on="MVT_ID_mvt", how="left", validate="one_to_one")[col].to_numpy(float)
    rows = {}
    for g, apts in GROUPS.items():
        m = keep & tr["ADEP_mvt"].isin(apts).to_numpy()
        for s, p in preds.items():
            e2 = (p[m] - y[m]) ** 2
            rows[(g, s)] = {"satır": int(m.sum()), "RMSE": np.sqrt(e2.mean()),
                            "kayma ağırlıklı RMSE": np.sqrt(np.average(e2, weights=w[m])),
                            "ağırlıklı ort. hata": np.average(p[m] - y[m], weights=w[m])}
    print("Satırlar: 24 saat (hedef ≥ 80.000) ve 24 saat ön koşulu hariç. Ağırlık: testin gecikme dağılımı / eğitiminki.")
    print(pd.DataFrame(rows).T.round(1).to_string())
    for g, apts in GROUPS.items():
        m = tr["ADEP_mvt"].isin(apts).to_numpy() & keep
        print(f"  {g}: ağırlık ort {w[m].mean():.2f}, en büyük {w[m].max():.2f}, etkin örnek {w[m].sum() ** 2 / (w[m] ** 2).sum():.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])

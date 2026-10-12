"""Model B adayları — 4 çeyrekli CV (her çeyrek diğer 9 ayla eğitilen modelle tahmin edilir; src/model_b.py).

Taban: v126_B = gönderimlerdeki Model B'nin yöntemi (PRC_4/src/train_model_b_production_grandmaster.py), aynı katlarda.

Eğitimde 22.470 NM'siz satır var; tek bölme (Ocak+Temmuz ~5.400 satır, 24 saatlik satırlar birkaç tane) çok gürültülü.
Ölçüler: OOF RMSE (tümü, 24 saat satırları hariç, LIRF / diğerleri, Ocak+Temmuz), çeyrek bazında RMSE.
Genel etki: testte Model B satırlarının payı %1,534; Model A RMSE'si 226,95 (B1, split_1) varsayılarak toplam RMSE.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

CANDIDATES = {
    "duz_havalimani": dict(mixture=False),
    "duz_havalimani_tavan3500": dict(mixture=False, cap_non_lirf=3500.0),
    "karisim_havalimani": dict(mixture=True),
    "karisim_global": dict(mixture=True, reg_scope="global"),
    "karisim_havalimani_tavan3500": dict(mixture=True, cap_non_lirf=3500.0),
}
SHARE_B = 5290 / 344841
RMSE_A = 226.951


def md_table(tab):
    head = "| Aday | " + " | ".join(tab.columns) + " |"
    sep = "|---|" + "---:|" * len(tab.columns)
    rows = [f"| {i} | " + " | ".join(f"{v:,.1f}" for v in r) + " |" for i, r in tab.iterrows()]
    return "\n".join([head, sep] + rows)


def summary(df, pred):
    y = df[MB.TARGET_B].to_numpy(float)
    e2 = (pred - y) ** 2
    m = lambda mask: float(np.sqrt(e2[mask].mean())) if mask.any() else float("nan")  # noqa: E731
    r = MB.regimes(df)
    lirf = df["ADEP_mvt"].eq("LIRF").to_numpy()
    jj = df["MVT_MONTH"].isin([1, 7]).to_numpy()
    allm = np.ones(len(df), bool)
    q = {name: m(np.isin(df["MVT_MONTH"], ms)) for name, ms in V.QUARTERS.items()}
    return {"rmse": m(allm), "rmse_24s_haric": m(r != 2), "lirf": m(lirf), "diger": m(~lirf), "ocak_temmuz": m(jj),
            "normal": m(r == 0), "yer_tutucu": m(r == 1), "s24": m(r == 2),
            "genel_rmse": float(np.sqrt((1 - SHARE_B) * RMSE_A ** 2 + SHARE_B * e2.mean())), **q}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    df = MB.load_b("train")
    feats = MB.features_b(df.columns)
    feats126 = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    print(f"Model B satırı {len(df):,} | özellik {len(feats)} | rejimler {np.bincount(MB.regimes(df)).tolist()}")
    folds = V.quarter_folds(df["MVT_MONTH"])
    oof = {"medyan": np.full(len(df), np.nan), "v126_B": np.full(len(df), np.nan),
           **{k: np.full(len(df), np.nan) for k in CANDIDATES}}
    for q, tr_m, va_m in folds:
        tr, va = df[tr_m].reset_index(drop=True), df[va_m].reset_index(drop=True)
        med = tr.groupby("ADEP_mvt")[MB.TARGET_B].median()
        oof["medyan"][va_m] = va["ADEP_mvt"].map(med).to_numpy(float)
        oof["v126_B"][va_m] = MB.fit_predict_v126b(tr, va, feats126)["taksi"]
        for name, kw in CANDIDATES.items():
            oof[name][va_m] = MB.fit_predict(tr, va, feats, **kw)["taksi"]
        print(f"  {q} tamam ({time.time() - t0:.0f} sn)", flush=True)
    rows = {k: summary(df, p) for k, p in oof.items()}
    tab = pd.DataFrame(rows).T
    print(tab.round(1).to_string())
    md = ["# Model B — 4 çeyrekli CV", "", f"Satır {len(df):,}; özellik {len(feats)}. RMSE (sn), taksi 60 sn alt sınırlı.",
          "`genel_rmse`: Model A RMSE'si 226,95 ve Model B payı %1,534 varsayımıyla toplam RMSE.", "",
          md_table(tab.round(1))]
    (config.LOG_DIR / "b_model_cv.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy(), "y": df[MB.TARGET_B].to_numpy(float), **oof}).to_parquet(
        config.DATA_DIR / "preds" / "BM_cv_oof.parquet", index=False)
    V.log_experiment({"ad": "BM_ceyrek_cv", "tur": "Model B", "ozellik_sayisi": len(feats),
                      "sonuc": tab.round(3).to_dict(orient="index"), "sure_sn": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()

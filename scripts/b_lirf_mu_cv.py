"""08.10: LIRF NM'siz satırlarında normal rejim μ'sünü yeni μ ile değiştirmenin CV'si. Üretimdeki LIRF formülü
(b_lirf_te.uretim): p0 · μ + p1 · max(gecikme, 60) + p2 · (gecikme > 40.000 ? gecikme : 87.400), ön koşullu satırlar
değişmez. p: b_lirf_te.fit_lirf (v126 + oran özellikleri) çeyrek CV'si; μ_eski: ilk tüm satırlı model (BM_cv_tum_oof,
üretimde TEST_B_tum_hepsi), μ_yeni: 0,5 · LightGBM + 0,5 · CatBoost (BM_cv_tum_dparkktahkomgspcyte_cb_oof).
Kullanım: python scripts/b_lirf_mu_cv.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config, model_b as MB, validation as V  # noqa: E402
import b_lirf_te as L  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ref = L.lirf_ham()
    df = L.b_satirlari("train")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    y = df[MB.TARGET_B].to_numpy(float)
    pre = np.asarray(MB.on_kosul_24s(df), bool)
    P = np.full((len(df), 3), np.nan)
    part1, part2 = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr, va = df[~va_m].reset_index(drop=True), df[va_m].reset_index(drop=True)
        ref_q = ref[~ref["ay"].isin(V.QUARTERS[q])]
        _, p, parts = L.fit_lirf(tr, va, feats, L.te_ozellik(tr, ref_q, oof=True), L.te_ozellik(va, ref_q, oof=False))
        P[va_m], part1[va_m], part2[va_m] = p, parts[1], parts[2]
    o = lambda f: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f), on="MVT_ID_mvt", how="left")  # noqa: E731
    mu_eski = o("BM_cv_tum_oof.parquet")["bilesen_mu"].to_numpy(float)
    M = o("BM_cv_tum_dparkktahkomgspcyte_cb_oof.parquet")
    mu_yeni = 0.5 * M["bilesen_mu"].to_numpy(float) + 0.5 * M["bilesen_mu_cb"].to_numpy(float)
    assert np.isfinite(mu_eski).all() and np.isfinite(mu_yeni).all()
    v126 = o("BM_cv_oof.parquet")["v126_B"].to_numpy(float)
    reg = np.where(y >= 40000, 2, np.where(y >= 2000, 1, 0))
    sat = {}
    secenek = [("μ eski (üretim)", mu_eski), ("μ yeni", mu_yeni)]
    nnf = config.DATA_DIR / "preds" / "BM_cv_mu_nn_dparkktahkomgspcyte_e12_t1_oof.parquet"
    if nnf.exists():  # 10.10: Model B μ'ye TabM (scripts/b_mu_nn.py); LIRF μ'de de aynı harman denenir
        mu_nn = o(nnf.name)["bilesen_mu_nn"].to_numpy(float)
        assert np.isfinite(mu_nn).all()
        secenek += [("μ yeni + TabM 0,3", 0.7 * mu_yeni + 0.3 * mu_nn), ("μ yeni + TabM 0,6", 0.4 * mu_yeni + 0.6 * mu_nn)]
    for ad, mu in secenek:
        pred = np.clip(P[:, 0] * mu + P[:, 1] * part1 + P[:, 2] * part2, 60.0, None)
        pred = np.where(pre, v126, pred)   # ön koşullu satırlar üretimde değişmez (v126)
        e2 = (pred - y) ** 2
        r = {"ön koşul dışı": np.sqrt(e2[~pre].mean()), "hepsi": np.sqrt(e2.mean()),
             "normal rejim (y < 2000)": np.sqrt(e2[~pre & (reg == 0)].mean()), "μ RMSE (normal)": np.sqrt(np.mean((mu - y)[reg == 0] ** 2))}
        for qn, ms in V.QUARTERS.items():
            m = df["MVT_MONTH"].isin(ms).to_numpy() & ~pre
            r[qn] = np.sqrt(e2[m].mean())
        sat[ad] = r
    print(f"LIRF B satırı {len(df):,}, ön koşul dışı {int((~pre).sum()):,}, normal rejim {int((reg == 0).sum()):,}")
    print(pd.DataFrame(sat).round(1).to_string())


if __name__ == "__main__":
    main()

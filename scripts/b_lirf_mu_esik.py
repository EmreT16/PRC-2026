"""10.10 gece: LIRF NM'siz satırlarında μ'ye TabM — yalnız normal rejim olasılığı yüksek satırlarda (p0 > t).
Gerekçe (b_lirf_mu_cv.py, 10.10 öğle): TabM'li μ normal rejimde çok daha isabetli (558 → 358) ama karışımda 4/4 çeyrek kötü;
eski μ'nün yukarı yanlılığı, kopya rejimi olup p0'ı yüksek görünmeyen satırlara karşı sigorta. TabM μ yalnız p0 > t satırlarda
kullanılırsa sigorta düşük p0'lı satırlarda korunur. μ_aday = p0 > t ise (1 − w) · μ_yeni + w · μ_TabM, değilse μ_yeni (üretim).
Önceden tanımlı kural: t ∈ {0,5; 0,7; 0,8; 0,9; 0,95}, w ∈ {0,3; 0,6}; iç içe çeyrek dışarıda seçim (3 çeyrekte en iyi (t, w) →
4.'de ölçüm) ön koşul dışı toplamda üretimden iyi VE tüm veriyle seçilen (t, w)'de ≥ 3/4 çeyrek iyi.
Kullanım: python scripts/b_lirf_mu_esik.py
"""
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
    mu_nn = o("BM_cv_mu_nn_dparkktahkomgspcyte_e12_t1_oof.parquet")["bilesen_mu_nn"].to_numpy(float)
    assert np.isfinite(mu_nn).all()
    Q = {q: df["MVT_MONTH"].isin(ms).to_numpy() & ~pre for q, ms in V.QUARTERS.items()}

    def tahmin(mu):
        pred = np.clip(P[:, 0] * mu + P[:, 1] * part1 + P[:, 2] * part2, 60.0, None)
        return np.where(pre, v126, pred)
    e0 = (tahmin(mu_yeni) - y) ** 2
    ad = {("üretim", 0, 0): e0}
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        for w in (0.3, 0.6):
            mu = np.where(P[:, 0] > t, (1 - w) * mu_yeni + w * mu_nn, mu_yeni)
            ad[("aday", t, w)] = (tahmin(mu) - y) ** 2
    sat = {}
    for k, e2 in ad.items():
        sat[f"t={k[1]} w={k[2]}" if k[0] == "aday" else "üretim"] = {
            "ön koşul dışı": np.sqrt(e2[~pre].mean()), **{q: np.sqrt(e2[m].mean()) for q, m in Q.items()},
            "p0>t satır": int(((P[:, 0] > k[1]) & ~pre).sum()) if k[0] == "aday" else 0}
    t_ = pd.DataFrame(sat).T
    print(f"LIRF B satırı {len(df):,}, ön koşul dışı {int((~pre).sum()):,}")
    print(t_.round(1).to_string())
    adaylar = [k for k in ad if k[0] == "aday"]
    sse_ic = 0.0
    for q, m in Q.items():
        diger = ~pre & ~m
        en = min(adaylar, key=lambda k: ad[k][diger].sum())
        sse_ic += ad[en][m].sum()
        print(f"  {q}: diğer çeyreklerle seçilen t = {en[1]}, w = {en[2]}")
    ic = np.sqrt(sse_ic / (~pre).sum()) - np.sqrt(e0[~pre].mean())
    en = min(adaylar, key=lambda k: ad[k][~pre].sum())
    q_iyi = int(sum(ad[en][m].mean() < e0[m].mean() for m in Q.values()))
    kabul = ic < 0 and q_iyi >= 3
    print(f"iç içe çeyrek seçimi: ön koşul dışı {ic:+.1f} | tüm veriyle en iyi t = {en[1]}, w = {en[2]} "
          f"({np.sqrt(ad[en][~pre].mean()) - np.sqrt(e0[~pre].mean()):+.1f}), iyi çeyrek {q_iyi}/4 → {'KABUL' if kabul else 'RET'}")


if __name__ == "__main__":
    main()

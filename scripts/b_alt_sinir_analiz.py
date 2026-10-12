"""11.10: denenen son işlem kuralı —
"taksi tahmini > 43.200 sn ise tahmin = max(tahmin, gecikme)" — eğitim verisi ve LIRF üretim düzeninin çeyrek dışı CV'siyle.
(1) Olgu: gecikmesi > 12 sa B satırlarında gerçek taksi hiç gecikmeden küçük değil mi? (2) LIRF üretim OOF'u (b_lirf_mu_esik.py
ile aynı kurulum: ön koşul dışı = fit_lirf karışımı, μ melez; ön koşullu = v126) üzerinde kuralın SSE etkisi, çeyrek çeyrek.
Üretimdeki 24 sa karışımı (v320) da çeyrek dışarıda uygulanır (karisimsiz → uygulanmaz).
LIRF dışı B satırlarında tavan 5.000 olduğu için kural tetiklenmez. Kullanım: python scripts/b_alt_sinir_analiz.py [karisimsiz]
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
import b_lirf_tarih_kaymasi as K  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ref = L.lirf_ham()
    df = L.b_satirlari("train")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    y, g = df[MB.TARGET_B].to_numpy(float), df[MB.DELAY].to_numpy(float)
    pre = np.asarray(MB.on_kosul_24s(df), bool)
    P = np.full((len(df), 3), np.nan)
    part1, part2 = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr, va = df[~va_m].reset_index(drop=True), df[va_m].reset_index(drop=True)
        ref_q = ref[~ref["ay"].isin(V.QUARTERS[q])]
        _, p, parts = L.fit_lirf(tr, va, feats, L.te_ozellik(tr, ref_q, oof=True), L.te_ozellik(va, ref_q, oof=False))
        P[va_m], part1[va_m], part2[va_m] = p, parts[1], parts[2]
    o = lambda f: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f), on="MVT_ID_mvt", how="left")  # noqa: E731
    M = o("BM_cv_tum_dparkktahkomgspcyte_cb_oof.parquet")
    mu = 0.5 * M["bilesen_mu"].to_numpy(float) + 0.5 * M["bilesen_mu_cb"].to_numpy(float)
    v126 = o("BM_cv_oof.parquet")["v126_B"].to_numpy(float)
    pred = np.where(pre, v126, np.clip(P[:, 0] * mu + P[:, 1] * part1 + P[:, 2] * part2, 60.0, None))
    print(f"LIRF B satırı {len(df):,} (ön koşullu {pre.sum()}); üretim OOF RMSE {np.sqrt(np.mean((pred - y) ** 2)):.1f}")
    if "karisimsiz" not in sys.argv[1:]:  # üretimdeki 24 sa karışımı (v320; b_lirf_tarih_kaymasi), p ve τ çeyrek dışarıda
        df["gec"], df["gece"] = g, g > df["LOCAL_MINOFDAY"].to_numpy(float) * 60
        df["kayma"] = (y >= 80000) & (np.abs(y - g) > 60)
        p24 = np.zeros(len(df))
        for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
            tr = df[~va_m]
            pred[va_m], p24[va_m] = K.uygula(df[va_m], pred[va_m], K.ESIK_SA, K.oran(tr, K.ESIK_SA), K.tau_hesap(tr))
        print(f"24 sa karışımı: {int((p24 > 0).sum())} satır; OOF RMSE {np.sqrt(np.mean((pred - y) ** 2)):.1f}")
    # 11.10: pencere=12,23 → kural yalnız gecikmesi bu aralıktaki (sa) satırlarda; pencere=12,23,tum → tahmin koşulu da yok
    pen = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("pencere=")), None)
    tet = pred > 43200
    if pen:
        tet = ((g >= float(pen[0]) * 3600) & (g <= float(pen[1]) * 3600)) & (tet if len(pen) < 3 else True)
        print(f"pencere: gecikme {pen[0]}–{pen[1]} sa" + (" (tahmin koşulu yok)" if len(pen) == 3 else " ve tahmin > 43.200"))
    yeni = np.where(tet, np.maximum(pred, g), pred)
    deg = tet & (yeni != pred)
    print(f"kural tetiklenen (tahmin > 43.200) {tet.sum()} satır, değişen (tahmin < gecikme) {deg.sum()}")
    x = df.loc[deg, ["MVT_ID_mvt", "MVT_MONTH"]].assign(gec_sa=g[deg] / 3600, y=y[deg], tahmin=pred[deg], yeni=yeni[deg],
                                                         p0=P[deg, 0], on_kosul=pre[deg])
    x["e2_fark_mn"] = ((x.yeni - x.y) ** 2 - (x.tahmin - x.y) ** 2) / 1e6
    x["grup"] = np.select([x.y < x.gec_sa * 3600 - 60, (x.y - x.gec_sa * 3600).abs() <= 60], ["normal", "yer tutucu"], "24 sa")
    print(x.round(3).to_string(index=False))
    e0, e1 = (pred - y) ** 2, (yeni - y) ** 2
    print(f"\nSSE farkı (milyon sn²): toplam {(e1 - e0).sum() / 1e6:+.2f} | iyileşen {int((e1 < e0).sum())}, kötüleşen {int((e1 > e0).sum())}")
    for q, ms in V.QUARTERS.items():
        m = df["MVT_MONTH"].isin(ms).to_numpy()
        print(f"  {q}: {(e1 - e0)[m].sum() / 1e6:+.2f}")
    print(f"LIRF RMSE {np.sqrt(e0.mean()):.2f} → {np.sqrt(e1.mean()):.2f}; test ölçeğinde (344.841 satır) MSE farkı "
          f"≈ {(e1 - e0).sum() / 344_841:+.2f} sn² (≈ {((e1 - e0).sum() / 344_841) / (2 * 218.5):+.4f} RMSE)")


if __name__ == "__main__":
    main()

"""LIRF Model B: v126'nın rejim olasılıkları korunur, yalnızca normal rejim tahmini tüm satırlı μ ile değişir.

Neden: tüm satırlı karışım (tum_mu, B satırlı rejim olasılıklarıyla) LIRF'te ön koşul dışı CV'de 3.187 → 3.079 veriyor,
ama testte farkın yarısı 3 satırdan geliyor ve biri 24 saat senaryosuna %23,5 ağırlık veriyor (tek satırda ±LB saniyeleri).
Burada v126'nın p0/p1/p2'si aynen kalır: tahmin = v126 + p0·(μ_tüm − pred0). Yüksek riskli rejim kararları değişmez.
Kullanım: python scripts/b_lirf_mu.py cv       → 4 çeyrekli CV (μ_tüm: BM_cv_tum_oof.parquet bilesen_mu, katlar aynı)
          python scripts/b_lirf_mu.py uretim   → 12 ay; testin LIRF ön koşul dışı satırları → TEST_B_lirf_mu.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

PRED = config.DATA_DIR / "preds"
V126 = config.PRC4_DIR / "submissions" / "strong-moose_v126.parquet"


def swap(c, mu):
    """v126 LIRF karışımında normal bileşeni μ ile değiştir (60 sn alt sınır, v126 gibi)."""
    return np.clip(c["p0"] * mu + c["p1"] * c["pred1"] + c["p2"] * c["pred2"], 60.0, None)


def cv():
    df = MB.load_b("train")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    mu = df[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED / "BM_cv_tum_oof.parquet")[["MVT_ID_mvt", "bilesen_mu"]],
                                  on="MVT_ID_mvt", how="left", validate="one_to_one")["bilesen_mu"].to_numpy(float)
    base = df[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED / "BM_cv_oof.parquet")[["MVT_ID_mvt", "v126_B"]],
                                    on="MVT_ID_mvt", how="left", validate="one_to_one")["v126_B"].to_numpy(float)
    lirf = df["ADEP_mvt"].eq("LIRF").to_numpy()
    pre = MB.on_kosul_24s(df)
    v126, new = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr, va = df[~va_m].reset_index(drop=True), df[va_m].reset_index(drop=True)
        c = MB.fit_predict_v126b(tr, va, feats)
        v126[va_m] = c["taksi"]
        li = va["ADEP_mvt"].eq("LIRF").to_numpy()
        tmp = c["taksi"].copy()
        tmp[li] = swap({k: v[li] for k, v in c.items()}, mu[va_m][li])
        new[va_m] = tmp
    print(f"yeniden kurulan v126_B, b_model_cv OOF'uyla en büyük fark: {np.nanmax(np.abs(v126 - base)):.3f}")
    y = df[MB.TARGET_B].to_numpy(float)
    rows = {}
    for name, ms in list(V.QUARTERS.items()) + [("Oca+Tem", (1, 7)), ("hepsi", tuple(range(1, 13)))]:
        mm = df["MVT_MONTH"].isin(ms).to_numpy()
        for grp, g in (("LIRF ön koşul dışı", lirf & ~pre), ("LIRF ön koşul", lirf & pre)):
            m = mm & g
            if m.sum():
                rows[(name, grp)] = {"satır": int(m.sum()), "v126_B": np.sqrt(np.mean((v126[m] - y[m]) ** 2)),
                                     "v126_B + μ_tüm": np.sqrt(np.mean((new[m] - y[m]) ** 2)),
                                     "ort. hata v126_B": np.mean(v126[m] - y[m]), "ort. hata yeni": np.mean(new[m] - y[m])}
    tab = pd.DataFrame(rows).T.round(1)
    print(tab.to_string())
    V.log_experiment({"ad": "BM_lirf_mu_cv", "tur": "Model B", "sonuc": {f"{a}|{b}": v for (a, b), v in
                                                                         tab.to_dict(orient="index").items()}})


def uretim():
    t0 = time.time()
    tr, rk = MB.load_b("train"), MB.load_b("rank")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    ap = rk[rk["ADEP_mvt"].eq("LIRF")].reset_index(drop=True)
    c = MB.fit_predict_v126b(tr, ap, feats)
    h = ap[["MVT_ID_mvt"]].merge(pd.read_parquet(PRED / "TEST_B_tum_hepsi.parquet")[["MVT_ID_mvt", "bilesen_mu"]],
                                 on="MVT_ID_mvt", how="left", validate="one_to_one")
    old = ap[["MVT_ID_mvt"]].merge(pd.read_parquet(V126), on="MVT_ID_mvt", how="left", validate="one_to_one")
    v126 = old["TAXITIME_SEC_mvt"].to_numpy(float)
    r = np.corrcoef(c["taksi"], v126)[0, 1]
    print(f"LIRF test B {len(ap)} | yeniden kurulan v126 ↔ gönderilen v126: korelasyon {r:.5f}, RMS fark "
          f"{np.sqrt(np.mean((c['taksi'] - v126) ** 2)):.1f}")
    # gönderilen v126 taban alınır; yalnızca normal bileşenin farkı eklenir: v126 + p0·(μ_tüm − pred0)
    taksi = np.clip(v126 + c["p0"] * (h["bilesen_mu"].to_numpy(float) - c["pred0"]), 60.0, None)
    pre = MB.on_kosul_24s(ap)
    out = pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": "LIRF", "taksi": taksi, "v126": v126,
                        "p0": c["p0"], "pred0": c["pred0"], "mu_tum": h["bilesen_mu"].to_numpy(float), "on_kosul": pre})
    out = out[~pre].reset_index(drop=True)
    out.to_parquet(PRED / "TEST_B_lirf_mu.parquet", index=False)
    d = out["taksi"] - out["v126"]
    print(f"ön koşul dışı {len(out)} satır | ort. yeni {out['taksi'].mean():.0f}, v126 {out['v126'].mean():.0f} | RMS fark "
          f"{np.sqrt((d ** 2).mean()):.0f} | |fark| en büyük {d.abs().max():.0f} | süre {time.time() - t0:.0f} sn")
    print("fark yüzdelikleri:", np.percentile(d, [1, 5, 25, 50, 75, 95, 99]).round(0).tolist())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"cv": cv, "uretim": uretim}[sys.argv[1]]()

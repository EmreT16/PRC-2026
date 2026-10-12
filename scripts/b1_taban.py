"""B1 — Taban: v126 YENİ_200 düzenekte (split_1 ve split_2).

Özellikler v126 hâliyle (134; metin → 0 hatası dahil), TE kat içinde (ay dışı OOF). Model v126'nın tri-modeli,
bölmenin eğitim aylarıyla bir kez eğitilir (models_v126.py). Tahminler data\\preds\\B1_<split>.parquet'e ham ofset
olarak yazılır; puan gönderimdeki 60 sn alt sınırla hesaplanır (validation.floor_offset). A2, B2 ve sonraki adaylar
aynı doğrulama satırlarında bunlarla karşılaştırılır.

Kullanım: python scripts/b1_taban.py          (eğitim + puan)
          python scripts/b1_taban.py --puan   (yalnızca kayıtlı tahminleri puanlar)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, models_v126 as M, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
COMPONENTS = ("lgb_s42", "lgb", "cb", "xgb", "blend")


def score(split):
    """Kayıtlı B1 tahminlerini 60 sn alt sınırla puanlar ve kütüğe yazar (eğitim yok)."""
    d = pd.read_parquet(PRED_DIR / f"B1_{split}.parquet")
    w = V.month_weights(V.SPLITS[split])
    nm = d["NM_TAXI_DIFF_SEC"].to_numpy(float)
    y = d["y"].to_numpy(float)
    pf = {k: V.floor_offset(d[k], nm) for k in COMPONENTS}
    metrics = {k: V.weighted_rmse(y, p, d["MVT_MONTH"], w) for k, p in pf.items()}
    raw = V.weighted_rmse(y, d["blend"], d["MVT_MONTH"], w)["weighted_rmse"]
    cut = float((pf["blend"] < d["blend"].to_numpy()).mean())
    apt_rmse = {a: round(float(np.sqrt(np.mean((pf["blend"][g] - y[g]) ** 2))), 2)
                for a, g in d.groupby("ADEP_mvt").indices.items()}
    print(f"[{split}] ağırlıklı RMSE (60 sn alt sınırlı): "
          + " | ".join(f"{k} {m['weighted_rmse']:.3f}" for k, m in metrics.items()))
    print(f"  alt sınırsız harman {raw:.3f}; alt sınıra takılan satır %{100 * cut:.2f}")
    print("  aylar (harman): " + ", ".join(f"{mo}: {v['rmse']:.3f}" for mo, v in metrics["blend"]["months"].items()))
    print("  havalimanı (harman): " + ", ".join(f"{a} {v}" for a, v in apt_rmse.items()), flush=True)
    V.log_experiment({"ad": f"B1_puan_{split}", "tur": "B1", "split": split, "alt_sinir": V.TAXI_FLOOR,
                      "agirlikli_rmse": {k: m["weighted_rmse"] for k, m in metrics.items()},
                      "alt_sinirsiz_harman": raw, "alt_sinira_takilan_pay": cut,
                      "aylar_harman": metrics["blend"]["months"], "havalimani_harman": apt_rmse,
                      "tahmin_dosyasi": str(PRED_DIR / f"B1_{split}.parquet")})


def train(split):
    t0 = time.time()
    train, val, vm = F.prepare_split(split)
    feats = F.feature_list(train.columns)
    assert len(feats) == 134, len(feats)
    y_val = val[config.TARGET].to_numpy(float)
    preds = {k: np.full(len(val), np.nan) for k in ("lgb_s42", "lgb", "cb", "xgb")}
    print(f"[{split}] eğitim {len(train):,} | doğrulama {len(val):,} | hazırlık {time.time() - t0:.0f} sn", flush=True)
    for apt in sorted(val["ADEP_mvt"].unique()):
        ta = time.time()
        tr_i = (train["ADEP_mvt"] == apt).to_numpy()
        va_i = (val["ADEP_mvt"] == apt).to_numpy()
        out = M.fit_predict(F.matrix(train[tr_i], feats), train.loc[tr_i, config.TARGET].to_numpy(float),
                            F.matrix(val[va_i], feats), feats)
        for k, v in out.items():
            preds[k][va_i] = v
        rm = " ".join(f"{k} {np.sqrt(np.mean((v - y_val[va_i]) ** 2)):.1f}" for k, v in out.items())
        print(f"  {apt}: eğitim {tr_i.sum():,} doğrulama {va_i.sum():,} | {rm} (alt sınırsız) | "
              f"{time.time() - ta:.0f} sn", flush=True)
    preds["blend"] = M.blend(preds)
    assert all(np.isfinite(v).all() for v in preds.values())
    pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": val["ADEP_mvt"].to_numpy(),
                  "MVT_MONTH": val["MVT_MONTH"].to_numpy(), "MVT_DAY": val["MVT_DAY"].to_numpy(),
                  "NM_TAXI_DIFF_SEC": val["NM_TAXI_DIFF_SEC"].to_numpy(), "y": y_val, **preds}
                 ).to_parquet(PRED_DIR / f"B1_{split}.parquet", index=False)
    V.log_experiment({"ad": f"B1_taban_{split}", "tur": "B1", "split": split,
                      "model": "v126 YENİ_200 (düzenek: bölmenin eğitim aylarıyla tek eğitim)", "ozellik": len(feats),
                      "tahmin_dosyasi": str(PRED_DIR / f"B1_{split}.parquet"), "sure_sn": round(time.time() - t0, 1)})
    print(f"[{split}] süre {time.time() - t0:.0f} sn", flush=True)


def main(splits=("split_1", "split_2")):
    sys.stdout.reconfigure(encoding="utf-8")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if "--puan" not in sys.argv:
            train(split)
        score(split)


if __name__ == "__main__":
    main(tuple(a for a in sys.argv[1:] if a.startswith("split_")) or ("split_1", "split_2"))

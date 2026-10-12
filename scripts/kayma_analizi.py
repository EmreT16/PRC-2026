"""09.10 akşam: 2025 → 2026 dağılım kayması (adversarial validation), Model A satırları.

Soru: G2 (CatBoost) / G2b 2025 CV'de 5/5 ay iyi, 2026 Oca / Tem LB'de kötü. Hangi özelliklerin dağılımı 2026'da değişti?
Yöntem: aynı aylar (Oca + Tem) 2025 eğitim ↔ 2026 test satırları; sınıflandırıcı "hangi yıl" sorusunu özelliklerden
öğrenmeye çalışır (LightGBM, 3 kat). AUC ≈ 0,5 → kayma yok; yüksek AUC'ye katkı veren özellikler = kayanlar. Yıl / tarih
kimliği taşıyan sütunlar (tatil adı, gün sayaçları) çıkarılır. Ayrıca her özellik için tek değişkenli KS istatistiği.
Girdi: taban tablolar (base_v126_*_v1) + modele giren ek özellik dosyaları (kuyruk, gecikme, wx, komşu, günlük, pist, opdi).
Kullanım: python scripts/kayma_analizi.py   → kutuk/kayma_analizi.log
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

EK = ["kuyruk_oz", "gecikme_oz", "gecikme_pist_oz", "wx_oz", "komsu_oz", "gunluk_oz", "pist_oz", "opdi_oz"]
ATLA = ("holiday", "MVT_DAY", "MVT_ID", "TAXITIME", "NM_OFFSET", "YEAR", "_DATE", "TIME_UTC", "KOM_T", "KOM_PIST")  # KOM_T: zaman damgası yardımcısı (özellik değil)


def yukle(bolum, n):
    d = pd.read_parquet(config.DATA_DIR / f"base_v126_{bolum}_v1.parquet")
    d = d[d["MVT_MONTH"].isin([1, 7]) & d["NM_TAXI_DIFF_SEC"].notna()]
    d = d.sample(min(n, len(d)), random_state=42)
    for f in EK:
        p = config.DATA_DIR / f"{f}.parquet"
        if p.exists():
            x = pd.read_parquet(p)
            x = x[x["MVT_ID_mvt"].isin(d["MVT_ID_mvt"])]
            d = d.merge(x, on="MVT_ID_mvt", how="left", suffixes=("", f"__{f}"))
    return d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = yukle("train", 150_000), yukle("rank", 150_000)
    ortak = [c for c in a.columns if c in b.columns and not any(s.lower() in c.lower() for s in ATLA)]
    num = [c for c in ortak if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    X = pd.concat([a[num], b[num]], ignore_index=True).astype(np.float32)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    print(f"2025 Oca/Tem {len(a):,} ↔ 2026 Oca/Tem {len(b):,} satır | sayısal özellik {len(num)}")
    # katlar GÜNE göre: gün düzeyindeki özellikler (günlük seriler, pist düzeni) her günü parmak izi gibi tanıtır; satır
    # bazlı katlarda aynı gün iki tarafta olunca AUC yapay olarak 1,0 çıkıyor (09.10 ilk koşu).
    gun = np.r_[2025 * 10000 + a["MVT_MONTH"].to_numpy() * 100 + a["MVT_DAY"].to_numpy(),
                2026 * 10000 + b["MVT_MONTH"].to_numpy() * 100 + b["MVT_DAY"].to_numpy()]
    kat = (pd.Series(gun).rank(method="dense").to_numpy().astype(int) * 7919) % 3
    oof, kazanc = np.zeros(len(y)), pd.Series(0.0, index=num)
    for k in range(3):
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4, seed=k),
                      lgb.Dataset(X[kat != k], y[kat != k]), 300)
        oof[kat == k] = m.predict(X[kat == k])
        kazanc += pd.Series(m.feature_importance("gain"), index=num)
    print(f"AUC (2025 ↔ 2026 ayırt etme): {roc_auc_score(y, oof):.4f}")
    kazanc = (kazanc / kazanc.sum()).sort_values(ascending=False)
    ks = {}
    for c in num:
        u, v = a[c].dropna().to_numpy(float), b[c].dropna().to_numpy(float)
        if len(u) > 100 and len(v) > 100:
            q = np.quantile(np.r_[u, v], np.linspace(0, 1, 201))
            fu = np.searchsorted(np.sort(u), q, "right") / len(u)
            fv = np.searchsorted(np.sort(v), q, "right") / len(v)
            ks[c] = float(np.max(np.abs(fu - fv)))
    ks = pd.Series(ks)
    tab = pd.DataFrame({"kazanc_payi": kazanc, "KS": ks, "ort_2025": a[num].mean(), "ort_2026": b[num].mean(),
                        "bos_2025": a[num].isna().mean(), "bos_2026": b[num].isna().mean()})
    print("\nEn çok ayırt eden 30 özellik (kazanç payına göre):")
    print(tab.sort_values("kazanc_payi", ascending=False).head(30).round(4).to_string())
    print("\nKS en yüksek 20 özellik:")
    print(tab.sort_values("KS", ascending=False).head(20).round(4).to_string())


if __name__ == "__main__":
    main()

"""A3 — Özellik parite testi.

Eğitim satırlarını (Ocak+Temmuz 2025) test satırlarıyla (Ocak+Temmuz 2026) karşılaştırır; 12 ay ↔ 2 ay
karşılaştırması mevsimsel yanlış alarm verirdi. Eşikler bilinçli olarak kaba: amaç gerçek dağılım kaymasını
değil, eğitim ve testte farklı çalışan kodu yakalamak.

- HATA: sütun bir tarafta yok; tür uyuşmuyor; testte sabit ama eğitimde değil; NaN oranı farkı > 5 puan.
- UYARI: iki tarafta da sabit (ölü sütun); standart ortalama farkı > 0,25; KS > 0,15;
  kategorikte test satırlarının > %5'i eğitimde hiç görülmemiş bir değer taşıyor.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NAN_PP = 0.05
SMD = 0.25
KS = 0.15
UNSEEN = 0.05


def _numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype)


def compare_features(train, test, feats, sample=100_000, seed=0):
    """Özellik başına parite ölçütleri ve seviye (HATA / UYARI / tamam) içeren tablo."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in feats:
        r = {"ozellik": c}
        if c not in train.columns or c not in test.columns:
            rows.append({**r, "seviye": "HATA", "neden": "sütun bir tarafta yok"})
            continue
        a, b = train[c], test[c]
        r.update(nan_egitim=float(a.isna().mean()), nan_test=float(b.isna().mean()),
                 benzersiz_egitim=int(a.nunique()), benzersiz_test=int(b.nunique()))
        hata, uyari = [], []
        if _numeric(a) != _numeric(b):
            hata.append(f"tür uyuşmuyor ({a.dtype} ↔ {b.dtype})")
        if abs(r["nan_egitim"] - r["nan_test"]) > NAN_PP:
            hata.append(f"NaN oranı {r['nan_egitim']:.1%} ↔ {r['nan_test']:.1%}")
        if r["benzersiz_test"] <= 1 < r["benzersiz_egitim"]:
            hata.append("testte sabit, eğitimde değil")
        elif r["benzersiz_test"] <= 1 and r["benzersiz_egitim"] <= 1:
            uyari.append("iki tarafta da sabit (ölü)")
        elif r["benzersiz_egitim"] <= 1 < r["benzersiz_test"]:
            uyari.append("eğitimde sabit, testte değil (model kullanamaz; biçim farkı?)")
        if _numeric(a) and _numeric(b) and r["benzersiz_egitim"] > 1:
            av, bv = a.to_numpy(float), b.to_numpy(float)
            r["sonsuz_egitim"], r["sonsuz_test"] = int(np.isinf(av).sum()), int(np.isinf(bv).sum())
            if r["sonsuz_test"] and not r["sonsuz_egitim"]:
                uyari.append(f"testte {r['sonsuz_test']} sonsuz değer")
            av, bv = av[np.isfinite(av)], bv[np.isfinite(bv)]
            if len(av) and len(bv):
                r["ort_egitim"], r["ort_test"] = float(av.mean()), float(bv.mean())
                r["smd"] = float(abs(bv.mean() - av.mean()) / (av.std() + 1e-9))
                r["ks"] = float(ks_2samp(rng.choice(av, min(sample, len(av)), replace=False),
                                         rng.choice(bv, min(sample, len(bv)), replace=False)).statistic)
                r["sifir_egitim"], r["sifir_test"] = float((av == 0).mean()), float((bv == 0).mean())
                if r["smd"] > SMD:
                    uyari.append(f"ortalama farkı {r['smd']:.2f} std")
                if r["ks"] > KS:
                    uyari.append(f"KS {r['ks']:.2f}")
        elif not _numeric(a):
            seen = set(a.dropna().astype(str).unique())
            r["gorulmemis_test"] = float((~b.dropna().astype(str).isin(seen)).mean()) if b.notna().any() else 0.0
            if r["gorulmemis_test"] > UNSEEN:
                uyari.append(f"test satırlarının {r['gorulmemis_test']:.1%}'i eğitimde görülmemiş değer")
        r["seviye"] = "HATA" if hata else ("UYARI" if uyari else "tamam")
        r["neden"] = "; ".join(hata + uyari)
        rows.append(r)
    return pd.DataFrame(rows)

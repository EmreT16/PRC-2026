"""A3 — v126 özellik hattının parite testi.

Nihai modelin göreceği hâliyle karşılaştırır: eğitim satırlarında TE 12 ay üzerinde ay dışı OOF,
test satırlarında 12 ayın tamamından. İki düzey:
  (1) ham değerler (v126 matrisine dönüşmeden önce),
  (2) v126 model matrisi (metin → 0 ve NaN → 0 hatalarıyla birlikte).
Çıktı: kutuk\\a3_parite.csv (tüm özellikler) ve kutuk\\a3_parite.md (özet).
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, parity as P, validation as V  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = F.base_table("train")
    rk = F.base_table("rank")
    train = tr[tr[config.TARGET].notna() & F.clean_mask(tr)].reset_index(drop=True)
    test = rk[rk["NM_TAXI_DIFF_SEC"].notna()].reset_index(drop=True)
    del tr, rk
    F.add_te(train, test, train["MVT_MONTH"].to_numpy())
    feats = F.feature_list(train.columns)
    tr17 = train[train["MVT_MONTH"].isin([1, 7])]
    print(f"eğitim Oca+Tem {len(tr17):,} | test {len(test):,} | özellik {len(feats)} | {time.time() - t0:.0f} sn")

    raw = P.compare_features(tr17, test, feats)
    mat = P.compare_features(F.matrix(tr17, feats), F.matrix(test, feats), feats)
    out = raw.merge(mat[["ozellik", "seviye", "neden"]], on="ozellik", suffixes=("_ham", "_matris"))
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.LOG_DIR / "a3_parite.csv", index=False, encoding="utf-8")

    lines = ["# A3 — v126 özellik hattının parite testi", "",
             f"Eğitim Ocak+Temmuz 2025 ({len(tr17):,} satır) ↔ test Ocak+Temmuz 2026 ({len(test):,} satır); "
             f"{len(feats)} özellik. Eşikler: `src/parity.py`.", ""]
    for level in ("ham", "matris"):
        sub = out[out[f"seviye_{level}"] != "tamam"]
        cnt = out[f"seviye_{level}"].value_counts().to_dict()
        lines += [f"## {'Ham değerler' if level == 'ham' else 'v126 model matrisi'}: {cnt}", "",
                  "| Özellik | Seviye | Neden |", "|---|---|---|"]
        lines += [f"| `{r.ozellik}` | {r[f'seviye_{level}']} | {r[f'neden_{level}']} |" for _, r in sub.iterrows()]
        lines.append("")
    (config.LOG_DIR / "a3_parite.md").write_text("\n".join(lines), encoding="utf-8")
    for level in ("ham", "matris"):
        print(f"\n[{level}] " + str(out[f"seviye_{level}"].value_counts().to_dict()))
        for _, r in out[out[f"seviye_{level}"] != "tamam"].iterrows():
            print(f"  {r[f'seviye_{level}']:5s} {r.ozellik:28s} {r[f'neden_{level}']}")
    V.log_experiment({"ad": "A3_parite_v126", "tur": "A3", "ozellik": len(feats),
                      "ham": out["seviye_ham"].value_counts().to_dict(),
                      "matris": out["seviye_matris"].value_counts().to_dict()})
    print(f"toplam {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()

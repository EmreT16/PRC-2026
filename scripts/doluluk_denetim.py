"""Özellik doluluk denetimi (07.10 gece; Y8 türü sessiz hataları yakalamak için).

Y8: düzelticiler 12 ayla eğitilirken mvt_epoch yalnız 6 ayı okuyordu → `yerel` eğitim satırlarının yarısında boştu,
testte doluydu; hiçbir uyarı yoktu. Bu betik her özellik dosyası için ay ay dolu (sonlu) satır payını ve sıfır payını
hesaplar ve iki tür sapmayı listeler:
  A) eğitim içi: 12 ayın dolu payları arasında fark > ESIK_AY (bir ay / ay grubu eksik okunmuş olabilir);
  B) eğitim ↔ test: aynı takvim ayında (Oca, Tem) 2025 ile 2026 dolu payı farkı > ESIK_TEST.
Her sapma hata değildir (gerçek kapsama değişimi de olabilir, ör. EDDF ADS-B); liste elle yorumlanır.
Kapsam: base_v126 sütunları, data/*.parquet yan dosyaları, data/adsb5 (günlük), opdi_oz.
Kullanım: python scripts/doluluk_denetim.py → kutuk/doluluk_denetim.csv ve ekrana özet.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

DATA = config.ROOT / "data"
ESIK_AY = 0.15
ESIK_TEST = 0.10


def anahtar():
    a = pd.read_parquet(DATA / "base_v126_train_v1.parquet", columns=["MVT_ID_mvt", "MVT_MONTH", "ADEP_mvt"])
    a["kume"] = "e"
    b = pd.read_parquet(DATA / "base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "MVT_MONTH", "ADEP_mvt"])
    b["kume"] = "t"
    k = pd.concat([a, b], ignore_index=True)
    k["grup"] = k["kume"] + k["MVT_MONTH"].astype(int).astype(str).str.zfill(2)
    return k


def olc(k, d, kaynak):
    m = k[["MVT_ID_mvt", "grup"]].merge(d, on="MVT_ID_mvt", how="left")
    satir = []
    for c in d.columns:
        if c == "MVT_ID_mvt" or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        x = m[c].to_numpy(float)
        dolu = pd.Series(np.isfinite(x)).groupby(m["grup"].to_numpy()).mean()
        sifir = pd.Series(x == 0).groupby(m["grup"].to_numpy()).mean()
        r = {"kaynak": kaynak, "ozellik": c}
        r.update({f"d_{g}": v for g, v in dolu.items()})
        r.update({f"z_{g}": v for g, v in sifir.items()})
        satir.append(r)
    return satir


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    k = anahtar()
    sat = []
    for f in ("base_v126_train_v1.parquet",):
        cols = [c for c in pd.read_parquet(DATA / f).columns]
        d = pd.concat([pd.read_parquet(DATA / f), pd.read_parquet(DATA / "base_v126_rank_v1.parquet")], ignore_index=True)
        sat += olc(k, d[[c for c in cols if c not in ("TAXITIME_SEC_mvt", "MVT_MONTH")]], "base_v126")
    for f in sorted(glob.glob(str(DATA / "*.parquet"))):
        ad = Path(f).name
        if ad.startswith("base_v126") or ad == "park_merkez.parquet":
            continue
        d = pd.read_parquet(f)
        if "MVT_ID_mvt" not in d.columns:
            continue
        sat += olc(k, d.drop_duplicates("MVT_ID_mvt"), ad)
    for kl in ("adsb5", "adsb5_s30", "adsb5_s10"):
        fs = sorted(glob.glob(str(DATA / kl / "oz_*.parquet")))
        if fs:
            sat += olc(k, pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates("MVT_ID_mvt"), kl)
    t = pd.DataFrame(sat)
    eg = [c for c in t.columns if c.startswith("d_e")]
    t["egitim_ay_fark"] = t[eg].max(1) - t[eg].min(1)
    t["egitim_en_dusuk_ay"] = t[eg].idxmin(1).str[3:]
    t["test_fark_oca"] = t.get("d_t01", np.nan) - t.get("d_e01", np.nan)
    t["test_fark_tem"] = t.get("d_t07", np.nan) - t.get("d_e07", np.nan)
    t.to_csv(config.LOG_DIR / "doluluk_denetim.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    print(f"{len(t)} özellik denetlendi → kutuk/doluluk_denetim.csv\n")
    a = t[t["egitim_ay_fark"] > ESIK_AY].sort_values("egitim_ay_fark", ascending=False)
    print(f"A) eğitim aylari arasında dolu payı farkı > {ESIK_AY}: {len(a)}")
    print(a[["kaynak", "ozellik", "egitim_ay_fark", "egitim_en_dusuk_ay"] + eg].round(2).to_string(index=False))
    b = t[(t["test_fark_oca"].abs() > ESIK_TEST) | (t["test_fark_tem"].abs() > ESIK_TEST)]
    print(f"\nB) 2025 ↔ 2026 aynı ay dolu payı farkı > {ESIK_TEST}: {len(b)}")
    print(b[["kaynak", "ozellik", "d_e01", "d_t01", "d_e07", "d_t07"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

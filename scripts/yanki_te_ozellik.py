"""Yankı geçmişi özellikleri (04.10; LIRF Temmuz analizi).
Gecikmeli kalkışta (kalkış − tarife > 40 dk) kaydedilen BLOCK'un tarife saatine eşit olma (±60 sn) oranı; anahtar başına
hedef kodlama (havalimanı içinde): uçuş kodu, havayolu, varış, park yeri. A ve B (NM'siz) satırları birlikte.
2025 satırları: ay dışı (o ay hariç 11 ay); 2026 test: 2025'in 12 ayı. Düzleştirme k = 20, havalimanı ortalamasına doğru.
Çıktı: data/yanki_te_oz.parquet (YTE_UCUS, YTE_HY, YTE_VARIS, YTE_PARK, YTE_N_UCUS)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

K = 20.0
ANAHTAR = {"YTE_UCUS": "FLIGHT_CODE", "YTE_HY": "AIRLINE_LETTERS", "YTE_VARIS": "ADES_mvt", "YTE_PARK": "STAND_mvt"}
SUT = ["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "TAKEOFF_VS_SCHED_SEC"] + list(ANAHTAR.values())
tr = pd.read_parquet(config.DATA_DIR / "base_v126_train_v1.parquet", columns=SUT + ["TAXITIME_SEC_mvt"])
te = pd.read_parquet(config.DATA_DIR / "base_v126_rank_v1.parquet", columns=SUT)
tr, te = tr[tr.MVT_ID_mvt.notna()].reset_index(drop=True), te[te.MVT_ID_mvt.notna()].reset_index(drop=True)
gec = (tr.TAKEOFF_VS_SCHED_SEC > 2400) & (tr.TAXITIME_SEC_mvt < 40000)
tr["yanki"] = ((tr.TAKEOFF_VS_SCHED_SEC - tr.TAXITIME_SEC_mvt).abs() <= 60).astype(float)
g = tr[gec]
print(f"2025 gecikmeli satır {len(g):,}; havalimanı yankı oranı %:",
      (g.groupby("ADEP_mvt").yanki.mean() * 100).round(1).to_dict(), flush=True)


def kodla(kaynak, hedef, ad, anahtar):
    s = kaynak.groupby(["ADEP_mvt", anahtar], observed=True).yanki.agg(["sum", "count"])
    m = kaynak.groupby("ADEP_mvt").yanki.mean()
    h = hedef[["ADEP_mvt", anahtar]].merge(s, left_on=["ADEP_mvt", anahtar], right_index=True, how="left")
    onc = hedef["ADEP_mvt"].map(m).to_numpy(float)
    cnt = h["count"].fillna(0).to_numpy(float)
    return (h["sum"].fillna(0).to_numpy(float) + K * onc) / (cnt + K), cnt


out_tr = pd.DataFrame({"MVT_ID_mvt": tr.MVT_ID_mvt})
for c in ANAHTAR:
    out_tr[c] = np.nan
out_tr["YTE_N_UCUS"] = np.nan
for ay in sorted(tr.MVT_MONTH.unique()):
    i = (tr.MVT_MONTH == ay).to_numpy()
    kay = g[g.MVT_MONTH != ay]
    for c, a in ANAHTAR.items():
        v, n = kodla(kay, tr[i], c, a)
        out_tr.loc[i, c] = v
        if c == "YTE_UCUS":
            out_tr.loc[i, "YTE_N_UCUS"] = n
out_te = pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt})
for c, a in ANAHTAR.items():
    v, n = kodla(g, te, c, a)
    out_te[c] = v
    if c == "YTE_UCUS":
        out_te["YTE_N_UCUS"] = n
o = pd.concat([out_tr, out_te], ignore_index=True)
assert o.MVT_ID_mvt.is_unique
o.astype({c: "float32" for c in o.columns if c != "MVT_ID_mvt"}).to_parquet(config.DATA_DIR / "yanki_te_oz.parquet", index=False)
print(len(o), "\n", o.drop(columns="MVT_ID_mvt").describe().T[["count", "mean", "std", "max"]].round(3).to_string())
print("eğitim/test ortalama YTE_UCUS:", round(float(out_tr.YTE_UCUS.mean()), 4), round(float(out_te.YTE_UCUS.mean()), 4))

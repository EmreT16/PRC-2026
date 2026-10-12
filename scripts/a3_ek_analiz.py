"""A3 ek analizleri: parite testinin işaretlediği üç konu.

1. Kimliklerin yıldan yıla taşınması: test satırlarının ne kadarı 2025'in 12 ayında hiç görülmemiş
   CALLSIGN / FLIGHT_CODE / ROUTE_PAIR / ADES taşıyor (C2 kararı için).
2. FLIGHT_CODE'un yalnızca rakamdan oluşan değerleri: eğitim ↔ test, havalimanı bazında.
3. RWY_DISPLACED_THRESH_FT dağılım farkı: havalimanı bazında dolu oranı ve ortalama.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr = F.base_table("train")
    rk = F.base_table("rank")
    tr = tr[tr[config.TARGET].notna()]
    rk = rk[rk["NM_TAXI_DIFF_SEC"].notna()]

    print("1) Testte, 2025'in 12 ayında hiç görülmemiş değer taşıyan satır payı (Model A):")
    for c in ("CALLSIGN_flt", "FLIGHT_CODE", "ROUTE_PAIR", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt",
              "AIRCRAFT_TYPE_mvt"):
        seen = set(tr[c].dropna().astype(str))
        seen17 = set(tr.loc[tr["MVT_MONTH"].isin([1, 7]), c].dropna().astype(str))
        b = rk[c].dropna().astype(str)
        print(f"   {c:22s} 12 ay: {(~b.isin(seen)).mean():6.1%} | yalnızca Oca+Tem 2025: {(~b.isin(seen17)).mean():6.1%}"
              f" | test benzersiz {b.nunique():,}")

    print("\n2) FLIGHT_CODE yalnızca rakam olan satır payı:")
    for name, d in (("eğitim", tr), ("test", rk)):
        dig = d["FLIGHT_CODE"].astype(str).str.fullmatch(r"\d+")
        by = dig.groupby(d["ADEP_mvt"]).mean()
        print(f"   {name:7s} genel {dig.mean():.2%} | havalimanı: "
              + ", ".join(f"{a} {v:.1%}" for a, v in by.items() if v > 0.001))
    ex = rk.loc[rk["FLIGHT_CODE"].astype(str).str.fullmatch(r"\d+"), ["ADEP_mvt", "FLIGHT_CODE", "CALLSIGN_flt",
                                                                    "AIRCRAFT_OPERATOR_flt"]].head(5)
    print("   test örnekleri:\n" + ex.to_string(index=False))

    print("\n3) RWY_DISPLACED_THRESH_FT (dolu oranı / ortalama), eğitim Oca+Tem ↔ test:")
    t17 = tr[tr["MVT_MONTH"].isin([1, 7])]
    a = t17.groupby("ADEP_mvt")["RWY_DISPLACED_THRESH_FT"].agg(dolu=lambda s: s.notna().mean(), ort="mean")
    b = rk.groupby("ADEP_mvt")["RWY_DISPLACED_THRESH_FT"].agg(dolu=lambda s: s.notna().mean(), ort="mean")
    print(a.join(b, lsuffix="_egitim", rsuffix="_test").round(2).to_string())


if __name__ == "__main__":
    main()

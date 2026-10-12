"""09.10: aksama günü özellikleri (etiketsiz) → aksama_oz.parquet.

Bulgu: bazı günlerde bir havalimanında NM verisi olmayan (Model B) kalkış sayısı medyanın 4–37 katına çıkıyor (final testi:
EHAM 2–6 Ocak 84–123 / gün, LEMD 5 Şub, EDDF / EDDM 19 Şub, EGLL 2 ve 27 Haz, LFPG 2 Haz …). O günlerde NM'siz satırlar
olağan uçuşlar (aksama / hava) — Model B'nin eğitimindeki "olağan NM'siz uçuş" nüfusundan farklı. Özellikler yalnız satır
sayımlarından (etiket yok), eğitim (2025) ve test (2026) için aynı tanımla:
  AK_B_GUN     havalimanı-gün NM'siz kalkış sayısı
  AK_B_KAT     AK_B_GUN / havalimanının 2025 gün başına NM'siz medyanı (en az 1)
  AK_B_PAY     havalimanı-gün NM'siz payı (NM'siz / tüm kalkış)
  AK_B_60      ±60 dk penceredeki NM'siz kalkış sayısı (kendisi hariç)
  AK_B_PAY_60  ±60 dk penceredeki NM'siz payı
Kullanım: [PRC_FINAL=1] python scripts/aksama_ozellik.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

K = ["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "MVT_DAY", "MVT_MINOFDAY_UTC", "NM_TAXI_DIFF_SEC"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr = pd.read_parquet(config.DATA_DIR / "base_v126_train_v1.parquet", columns=K).assign(yil=2025)
    rk = pd.read_parquet(config.DATA_DIR / "base_v126_rank_v1.parquet", columns=K).assign(yil=2026)
    d = pd.concat([tr, rk], ignore_index=True).drop_duplicates("MVT_ID_mvt")
    d["B"] = d["NM_TAXI_DIFF_SEC"].isna().astype(float)
    d["gun"] = d["yil"] * 10000 + d["MVT_MONTH"] * 100 + d["MVT_DAY"]
    g = d.groupby(["ADEP_mvt", "gun"]).agg(AK_B_GUN=("B", "sum"), n=("B", "size")).reset_index()
    med = g[g.gun < 20260000].groupby("ADEP_mvt").AK_B_GUN.median().clip(lower=1.0)
    g["AK_B_KAT"] = g.AK_B_GUN / g.ADEP_mvt.map(med)
    g["AK_B_PAY"] = g.AK_B_GUN / g.n
    d = d.merge(g.drop(columns="n"), on=["ADEP_mvt", "gun"], how="left")
    # ±60 dk penceresi: gün içinde dakika sırasına göre kümülatif sayım (gece yarısını aşan pencere kesilir)
    d = d.sort_values(["ADEP_mvt", "gun", "MVT_MINOFDAY_UTC"]).reset_index(drop=True)
    b60, n60 = np.zeros(len(d)), np.zeros(len(d))
    for _, i in d.groupby(["ADEP_mvt", "gun"]).indices.items():
        t = d["MVT_MINOFDAY_UTC"].to_numpy(float)[i]
        b = d["B"].to_numpy(float)[i]
        cb = np.concatenate([[0.0], np.cumsum(b)])
        lo, hi = np.searchsorted(t, t - 60, "left"), np.searchsorted(t, t + 60, "right")
        b60[i] = cb[hi] - cb[lo] - b
        n60[i] = (hi - lo) - 1
    d["AK_B_60"], d["AK_B_PAY_60"] = b60, np.where(n60 > 0, b60 / np.maximum(n60, 1), 0.0)
    out = d[["MVT_ID_mvt", "AK_B_GUN", "AK_B_KAT", "AK_B_PAY", "AK_B_60", "AK_B_PAY_60"]]
    out.to_parquet(config.DATA_DIR / "aksama_oz.parquet", index=False)
    x = d[d.B == 1]
    print(f"aksama_oz: {len(out):,} satır | NM'siz satırlarda AK_B_KAT yüzdelikleri (2025 / 2026):")
    for y in (2025, 2026):
        print(y, np.percentile(x.loc[x.yil == y, "AK_B_KAT"], [50, 90, 99]).round(2))
    print("2025'te AK_B_KAT ≥ 4 olan NM'siz satır:", int(((x.yil == 2025) & (x.AK_B_KAT >= 4)).sum()),
          "| 2026:", int(((x.yil == 2026) & (x.AK_B_KAT >= 4)).sum()))


if __name__ == "__main__":
    main()

"""EUROCONTROL aylık havalimanı taksi-out verisi → 10 havalimanı tablosu (05.10; Y1 kalibrasyonu, scripts/g_kalibre.py).

Kaynak: ansperformance.eu veri sayfası (https://www.ansperformance.eu/data/), "Taxi-out additional time" (Excel, sürüm
15.09.2026, Oca 2018 – Tem 2026), kopyası extra_data/ansperf/txout.xlsx; "DATA" sayfası.
Sütunlar: TF (toplam kalkış), VALID_FL (geçerli uçuş: taksi > 120 dk, AOBT sonrası buz çözme ve eksik veri hariç),
TOTAL_REF_NB_FL (referans süresi olan uçuş), TOTAL_REF_TIME_MIN, TOTAL_ADD_TIME_MIN.
yay_ort = (TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN) · 60 / TOTAL_REF_NB_FL (sn; yayımlanan ortalama taksi-out süresi).
Çıktı: extra_data/ansperf/txout_10apt.parquet (2025–2026)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

KLASOR = config.EK_VERI / "ansperf"
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = pd.read_excel(KLASOR / "txout.xlsx", sheet_name="DATA")
    d = d[d.APT_ICAO.isin(APT) & d.YEAR.isin([2025, 2026])].sort_values(["APT_ICAO", "YEAR", "MONTH_NUM"]).reset_index(drop=True)
    d["yay_ort"] = (d.TOTAL_REF_TIME_MIN + d.TOTAL_ADD_TIME_MIN) * 60 / d.TOTAL_REF_NB_FL
    d.to_parquet(KLASOR / "txout_10apt.parquet", index=False)
    print(len(d), "satır;", d.groupby("YEAR").yay_ort.count().to_dict(), "dolu ay")

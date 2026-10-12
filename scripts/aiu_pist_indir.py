"""AIU havalimanı panosundan aylık pist taksi-out süreleri (05.10; Y1 pist katmanı).

Kaynak: EUROCONTROL Aviation Intelligence Unit havalimanı panosu, https://aiu-airport-dashboard.netlify.app/<icao>
(ansperformance.eu/dashboard/stakeholder/airport/db/<icao> buraya yönlenir). "Monthly Taxi-Out Time per Runway (Source:
APDF)" bölümündeki plotly grafiğinin sayfaya gömülü verisi okunur: 12 alt grafik = 12 ay, her birinde "Avg. Reference time" ve
"Avg. Additional time" izleri (dk / kalkış), x = pist, customdata = yıl.
Çıktı: extra_data/aiu/aiu_pist_aylik.parquet (apt, yil, ay, pist, add, ref, taksi_sn = (ref + add) · 60) ve
indirilen sayfalar aynı klasörde (<icao>.html).
Kullanım: python scripts/aiu_pist_indir.py [yerel]   (yerel: indirmeden klasördeki sayfaları ayrıştır)"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

KLASOR = config.EK_VERI / "aiu"
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
URL = "https://aiu-airport-dashboard.netlify.app/{}"
BOLUM = "Monthly Taxi-Out Time per Runway"
WIDGET = re.compile(r'<script type="application/json" data-for="([^"]+)">(.*?)</script>', re.S)


def indir(a):
    with urllib.request.urlopen(URL.format(a.lower()), timeout=120) as r:
        (KLASOR / f"{a.lower()}.html").write_bytes(r.read())


def ayristir(a):
    h = (KLASOR / f"{a.lower()}.html").read_text(encoding="utf-8")
    i = h.find(BOLUM)
    assert i >= 0, f"{a}: '{BOLUM}' bölümü yok"
    w = next(m for m in WIDGET.finditer(h) if m.start() > i)
    sat = []
    for t in json.loads(w.group(2))["x"]["data"]:
        ax = t.get("xaxis", "x")
        ay = 1 if ax == "x" else int(ax[1:])
        bil = "ref" if "Reference" in t["name"] else "add"
        sat += [(a, int(yil), ay, str(p), bil, v) for p, yil, v in zip(t["x"], t["customdata"], t["y"])]
    return sat


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    KLASOR.mkdir(parents=True, exist_ok=True)
    if "yerel" not in sys.argv[1:]:
        for a in APT:
            indir(a)
    df = pd.DataFrame([s for a in APT for s in ayristir(a)], columns=["apt", "yil", "ay", "pist", "bil", "dk"])
    w = df.pivot_table(index=["apt", "yil", "ay", "pist"], columns="bil", values="dk", aggfunc="first").reset_index()
    w["taksi_sn"] = (w["ref"] + w["add"]) * 60
    w.to_parquet(KLASOR / "aiu_pist_aylik.parquet", index=False)
    print(len(w), "satır;", w.groupby("apt").size().to_dict())
    print("2026 ay kapsamı:", sorted(w[w.yil == 2026].ay.unique().tolist()))

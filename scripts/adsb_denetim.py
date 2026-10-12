"""adsb.lol günleri için eksiksizlik denetimi: günlüklerdeki iz sayısı / arşiv boyutu ve kalkış eşleşme oranı.

Neden: 29.01.2026 staging arşivinde bir bozukluk tar akışını sessizce erken bitirdi (33 bin iz; komşu günler 46–69 bin).
adsb_cikar.py artık bunu yakalıyor, ama önceki sürümle işlenmiş günler bu denetimle taranır.
Şüpheli: GB başına iz sayısı ay medyanının %75'inin altında ya da eşleşme oranı ay medyanının 10 puan altında.
Kullanım: python scripts/adsb_denetim.py   → şüpheli günleri yazar
"""
import re
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

HAM = Path(os.environ.get("PRC_ADSB_KOK", str(Path(__file__).resolve().parents[2] / "PRC_adsb"))) / "ham"
ADSB = config.DATA_DIR / "adsb"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    iz = {}
    for f in (config.ROOT / "kutuk").glob("adsb_*.log"):
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"(\d{4}-\d{2}-\d{2}): iz ([\d,]+)", ln)
            if m:
                iz[m.group(1)] = int(m.group(2).replace(",", ""))  # son kayıt geçerli
    rows = []
    for g, n in sorted(iz.items()):
        gb = sum(p.stat().st_size for p in (HAM / g).glob("*.tar*")) / 1e9
        oz = ADSB / f"oz_{g}.parquet"
        es = float((pd.read_parquet(oz, columns=["ADSB_ESLESME"])["ADSB_ESLESME"] > 0).mean()) if oz.exists() else None
        rows.append({"gün": g, "iz": n, "GB": round(gb, 2), "iz/GB": n / gb if gb else None, "eşleşme": es})
    d = pd.DataFrame(rows)
    d["ay"] = d["gün"].str[:7]
    d["iz/GB oranı"] = d["iz/GB"] / d.groupby("ay")["iz/GB"].transform("median")
    d["eşleşme farkı"] = d["eşleşme"] - d.groupby("ay")["eşleşme"].transform("median")
    sup = d[(d["iz/GB oranı"] < 0.75) | (d["eşleşme farkı"] < -0.10)]
    print(f"denetlenen gün {len(d)} | şüpheli {len(sup)}")
    print(sup.round(3).to_string(index=False) if len(sup) else "şüpheli gün yok")
    print(d.groupby("ay").agg(gun=("gün", "size"), iz_medyan=("iz", "median"), eslesme_medyan=("eşleşme", "median"))
          .round(3).to_string())


if __name__ == "__main__":
    main()

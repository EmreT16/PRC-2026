"""Final aşaması: data_final/ klasörüne yalnız eğitim tarafı (2025) ve sabit ara ürünleri kopyalar (08.10).

Neden: PRC_FINAL=1 iken tüm betikler config.DATA_DIR = data_final ile çalışır. Test tarafına dokunan her şey final_ranking'den
yeniden üretilir; eğitim satırlarına ait olup testten bağımsız olanlar (CV / OOF tahminleri, 2025 ADS-B gün özellikleri ve
seyrek kopyaları) ile sabit tablolar (park merkezleri, IATA → ICAO) değişmez → kopyalanır. Test dosyaları (TEST_*),
2026 ADS-B günleri ve eğitim + test birlikte hesaplanan özellik tabloları kopyalanmaz.
Kullanım: python scripts/final_hazirlik.py
"""
import csv
import shutil
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
KAY, HED = KOK / "data", KOK / "data_final"


def kopya(kaynak, hedef):
    hedef.parent.mkdir(parents=True, exist_ok=True)
    if not hedef.exists() or hedef.stat().st_size != kaynak.stat().st_size:
        shutil.copy2(kaynak, hedef)
        return 1
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(KOK / "docs" / "training_side_manifest.csv", encoding="utf-8") as f:
        eksik = [r["file"] for r in csv.DictReader(f) if not (KAY / r["file"]).exists()]
    assert not eksik, f"eğitim tarafı dosyası eksik ({len(eksik)}; docs/training_side_manifest.csv): {eksik[:5]}"
    n = {}
    n["preds (OOF / CV)"] = sum(kopya(p, HED / "preds" / p.name) for p in (KAY / "preds").glob("*.parquet")
                                if not p.name.startswith("TEST_"))
    n["adsb5 2025"] = sum(kopya(p, HED / "adsb5" / p.name) for p in (KAY / "adsb5").glob("oz_2025-*.parquet"))
    for d in ("adsb5_s10", "adsb5_s30"):
        assert not list((KAY / d).glob("oz_2026-*")), f"{d}: 2026 günü beklenmiyordu"
        n[d] = sum(kopya(p, HED / d / p.name) for p in (KAY / d).glob("*.parquet"))
    n["sabit"] = sum(kopya(KAY / f, HED / f) for f in ("park_merkez.parquet", "iata_icao.json"))
    print("kopyalanan (yeni ya da değişmiş):", n)
    print("data_final:", sorted(p.name for p in HED.iterdir()))


if __name__ == "__main__":
    main()

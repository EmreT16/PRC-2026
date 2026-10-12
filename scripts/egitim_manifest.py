"""12.10: final hattının hazır aldığı eğitim tarafı (2025) dosyalarının manifesti.
scripts/final_hazirlik.py bu dosyaları data/ → data_final/ kopyalar (TEST_ olmayan tahminler, 2025 ADS-B gün özellikleri ve
seyreltilmiş kopyaları, park merkezleri, IATA → ICAO); final_tum.sh bunları yeniden üretmez. Manifest her dosyanın boyutunu,
MD5'ini ve ön ekine göre üreten betiği yazar → docs/training_side_manifest.csv.
Kullanım: python scripts/egitim_manifest.py
"""
import csv
import hashlib
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
KAY = KOK / "data"
URETICI = [  # (ön ek, üreten betik)
    ("ADAY_", "scripts/aday_dogrula.py (component definitions in src/aday.py)"),
    ("GCB_dw", "scripts/gcb_dw_tarama.py"),
    ("GCB_", "scripts/gcb_varyant.py, scripts/gcb_ortalama.py"),
    ("NN_", "scripts/nn_bilesen.py"),
    ("ADSB_duz_cv", "scripts/adsb_duzeltici.py cv"),
    ("ADSB_B5", "scripts/adsb_b_duzeltici5.py cv"),
    ("BM_cv_mu_nn", "scripts/b_mu_nn.py"),
    ("BM_cv", "scripts/b_model_cv.py, scripts/b_model_cv_tum.py"),
    ("G2_cv", "scripts/gecikme_asama2.py"),
    ("IKINCI_cv", "scripts/ikinci_asama.py, scripts/komsu_ikinci.py"),
    ("LTFM_tarama", "scripts/ltfm_tarama.py"),
    ("B_l2", "scripts/b_l2_asama.py"),
    ("B_lirf", "scripts/b_lirf_deney.py"),
]


def uretici(ad):
    for on, b in URETICI:
        if ad.startswith(on):
            return b
    return "experiment output (copied, not read by the final pipeline as far as known)"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sat = []
    for p in sorted((KAY / "preds").glob("*.parquet")):
        if not p.name.startswith("TEST_"):
            sat.append(("preds", p, uretici(p.name)))
    for p in sorted((KAY / "adsb5").glob("oz_2025-*.parquet")):
        sat.append(("adsb5", p, "scripts/adsb_ozellik.py (2025 day features)"))
    for d in ("adsb5_s10", "adsb5_s30"):
        for p in sorted((KAY / d).glob("*.parquet")):
            sat.append((d, p, "scripts/adsb_ozellik.py (thinned-track copies, 10 % / 30 % of points)"))
    for f in ("park_merkez.parquet", "iata_icao.json"):
        sat.append(("fixed", KAY / f, "scripts/adsb_ozellik.py / src/adsb.py (stand centres; IATA → ICAO prefixes)"))
    cik = KOK / "docs" / "training_side_manifest.csv"
    cik.parent.mkdir(exist_ok=True)
    with open(cik, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "file", "bytes", "md5", "producer"])
        for g, p, u in sat:
            w.writerow([g, p.relative_to(KAY).as_posix(), p.stat().st_size, md5(p), u])
    top = sum(p.stat().st_size for _, p, _ in sat)
    print(f"{len(sat)} dosya, {top / 1e9:.2f} GB → {cik}")
    for g in ("preds", "adsb5", "adsb5_s10", "adsb5_s30", "fixed"):
        print(g, sum(1 for x in sat if x[0] == g))


if __name__ == "__main__":
    main()

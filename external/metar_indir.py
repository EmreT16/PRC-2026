"""IEM ASOS'tan 10 istasyonun ham METAR'ını ay ay indirir (2025-01…2025-12, 2026-01, -02, -06, -07).

Çıktı: ../metar_ham/metar_<ICAO>_<YYYYMM>.csv ve ../metar_ham/indirme_kaydi.csv
Kaldığı yerden devam eder: geçerli dosyası olan ay atlanır. Boş / eksik / hatalı yanıt artan beklemeyle yeniden denenir.
Geçerlilik: ilk satır 'station,valid' ile başlar ve satır sayısı ayın saat sayısının en az %50'si kadardır
(Avrupa istasyonları yarım saatte bir rapor verir; normalde ~2 × saat sayısı).
"""
import csv
import datetime as dt
import sys
import time
import urllib.request
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "metar_ham"
KAYIT = HAM / "indirme_kaydi.csv"
ISTASYONLAR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
AYLAR = [(2025, m) for m in range(1, 13)] + [(2026, 1), (2026, 2), (2026, 6), (2026, 7)]  # 08.10: final aşaması + Şub, Haz
ALANLAR = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "wxcodes", "metar"]
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
BEKLE = 5          # istekler arası (sn)
DENEME = 5         # ay başına en çok deneme


def ay_sonu(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def saat_sayisi(y, m):
    y2, m2 = ay_sonu(y, m)
    return int((dt.datetime(y2, m2, 1) - dt.datetime(y, m, 1)).total_seconds() // 3600)


def url(icao, y, m):
    y2, m2 = ay_sonu(y, m)
    q = [("station", icao)] + [("data", a) for a in ALANLAR] + [
        ("year1", y), ("month1", m), ("day1", 1), ("year2", y2), ("month2", m2), ("day2", 1),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"), ("missing", "M"),
        ("trace", "T"), ("direct", "no"), ("report_type", 3), ("report_type", 4)]
    return URL + "?" + "&".join(f"{k}={v}" for k, v in q)


def gecerli(yol, y, m):
    if not yol.exists() or yol.stat().st_size < 1000:
        return False, 0
    with open(yol, encoding="utf-8", errors="replace") as f:
        ilk = f.readline()
        n = sum(1 for _ in f)
    return ilk.startswith("station,valid") and n >= 0.5 * saat_sayisi(y, m), n


def main():
    yeni = not KAYIT.exists()
    with open(KAYIT, "a", newline="", encoding="utf-8") as kf:
        w = csv.writer(kf)
        if yeni:
            w.writerow(["zaman_utc", "icao", "ay", "satir", "deneme", "durum"])
        for icao in ISTASYONLAR:
            for y, m in AYLAR:
                yol = HAM / f"metar_{icao}_{y}{m:02d}.csv"
                ok, n = gecerli(yol, y, m)
                if ok:
                    continue
                for d in range(1, DENEME + 1):
                    try:
                        req = urllib.request.Request(url(icao, y, m), headers={"User-Agent": "PRC2026-arastirma/1.0"})
                        with urllib.request.urlopen(req, timeout=180) as r:
                            veri = r.read()
                        yol.write_bytes(veri)
                        ok, n = gecerli(yol, y, m)
                    except Exception as e:  # ağ hatası: yeniden dene
                        ok, n = False, 0
                        print(icao, y, m, "hata:", e, file=sys.stderr)
                    durum = "tamam" if ok else "gecersiz"
                    w.writerow([dt.datetime.utcnow().isoformat(timespec="seconds"), icao, f"{y}-{m:02d}", n, d, durum])
                    kf.flush()
                    print(icao, f"{y}-{m:02d}", n, durum, flush=True)
                    time.sleep(BEKLE * (1 if ok else 2 ** d))
                    if ok:
                        break


if __name__ == "__main__":
    main()

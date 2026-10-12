"""09.10: tercih edilen kopyası bozuk günleri doğrudan öbür kopyadan (prod ↔ staging) indirip çıkarır.

adsb_hat.py'nin içindeki "öbür kopya" denemesi Windows'ta başarısız oluyor: bozuk kopyanın çıkarılması yarıda kesilince
arşiv dosyası hâlâ açık kalıyor, silinemiyor (WinError 32). Burada her gün ayrı süreçte, hiçbir dosya açık değilken
adsb_indir.indir(g, diger=True) çağrılır (tercih edilen kopyayı atlayıp öbürünü indirir), sonra adsb_cikar.gun_isle.
Kullanım: python scripts/adsb_diger_kopya.py 2026-06-07 2026-06-23 2026-06-25 [isci=6]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_cikar as C  # noqa: E402
import adsb_indir as I  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    with Pool(int(opt.get("isci", 6))) as havuz:
        for g in I.gunler([a for a in sys.argv[1:] if "=" not in a]):
            d = I.HAM / g.isoformat()
            for p in list(d.glob("*.tar*")) + [d / ".tamam"]:
                p.unlink(missing_ok=True)
            I.indir(g, diger=True)
            C.gun_isle(g.isoformat(), havuz)
            assert (C.NOKTA / f"{g.isoformat()}.parquet").exists(), g
            for p in d.glob("*.tar*"):
                p.unlink()
            print(f"{g}: öbür kopyadan tamam", flush=True)

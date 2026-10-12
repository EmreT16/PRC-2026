"""adsb.lol hattı: her gün için indir (adsb_indir.py) → noktaları çıkar (adsb_cikar.py). İşlenmiş gün atlanır.

Kullanım: python scripts/adsb_hat.py <gün|aralık> [...] [isci=6]
  ör. python scripts/adsb_hat.py 2026-01-01:2026-01-31 2026-07-01:2026-07-31 2025-12-31 2026-06-30
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
    gunler = I.gunler([a for a in sys.argv[1:] if "=" not in a])
    I.HAM.mkdir(parents=True, exist_ok=True)
    with Pool(int(opt.get("isci", 6))) as havuz:
        for g in gunler:
            if (C.NOKTA / f"{g.isoformat()}.parquet").exists():
                print(f"{g}: noktalar zaten var", flush=True)
                continue
            for deneme in range(3):
                try:
                    I.indir(g)
                    break
                except Exception as e:  # ağ hatası: tekrar dene
                    print(f"{g}: indirme hatası ({e}), deneme {deneme + 1}", flush=True)
            else:
                continue
            try:
                C.gun_isle(g.isoformat(), havuz)
            except Exception as e:
                print(f"{g}: çıkarma hatası ({e}); öbür kopya deneniyor", flush=True)
                try:
                    I.indir(g, diger=True)
                    C.gun_isle(g.isoformat(), havuz)
                except Exception as e2:
                    print(f"{g}: öbür kopyada da çıkarma hatası ({e2})", flush=True)
            if opt.get("sil") == "1" and (C.NOKTA / f"{g.isoformat()}.parquet").exists():
                for p in (I.HAM / g.isoformat()).glob("*.tar*"):  # disk: noktalar çıkarıldıysa ham arşivi sil
                    p.unlink()

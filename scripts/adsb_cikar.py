"""adsb.lol günlük arşivinden 10 havalimanının çevresindeki alçak/yer noktalarını çıkarır.

Girdi: PRC_ADSB_KOK/ham/<gün>/*.tar.* (adsb_indir.py). Parçalar diske birleştirilmeden tek akış olarak okunur.
Arşivdeki traces/**/trace_full_<hex>.json dosyaları gzip'li JSON (biçim: readsb README-json.md "trace jsons"):
  nokta = [sn_ofset, lat, lon, irtifa_ft | "ground" | null, yer_hızı_kt, rota, bayraklar, dikey_hız, uçak_nesnesi|null,
           kaynak, geo_irtifa, geo_dikey_hız, ias, yatış]
Tutulan: havalimanı referans noktasının ±~11 km kutusunda, "ground" ya da irtifa ≤ rakım + 3.000 ft olan noktalar.
Çağrı kodu uçak nesnelerinden ileri doldurulur. Çıktı: PRC_ADSB_KOK/noktalar/<gün>.parquet
Kullanım: python scripts/adsb_cikar.py 2025-07-15 [2026-07-14 ...] [isci=10]
"""
import datetime as dt
import gzip
import io
import os
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

KOK = Path(os.environ.get("PRC_ADSB_KOK", str(Path(__file__).resolve().parents[2] / "PRC_adsb")))  # 02.10: D:/PRC_adsb ve C:/PRC_adsb buraya birleştirildi
HAM = KOK / "ham"
NOKTA = KOK / "noktalar"
# Havalimanı referans noktası (derece) ve rakım (ft); yaklaşık AIP değerleri
AP = {"EDDF": (50.0333, 8.5706, 364), "EDDM": (48.3538, 11.7861, 1487), "EGLL": (51.4775, -0.4614, 83),
      "EHAM": (52.3081, 4.7642, -11), "LEBL": (41.2971, 2.0785, 12), "LEMD": (40.4719, -3.5626, 2000),
      "LFPG": (49.0097, 2.5479, 392), "LIRF": (41.8003, 12.2389, 13), "LSZH": (47.4647, 8.5492, 1416),
      "LTFM": (41.2753, 28.7519, 325)}
DLAT = 0.10
DLON = {k: 0.10 / np.cos(np.radians(v[0])) for k, v in AP.items()}
UST = 3000.0
PARTI = 1500


class Zincir(io.RawIOBase):
    """Bölünmüş tar parçalarını tek akış gibi okur."""

    def __init__(self, yollar):
        self.fs = [open(p, "rb") for p in yollar]
        self.i = 0
        self.okunan = 0
        self.toplam = sum(Path(p).stat().st_size for p in yollar)

    def readable(self):
        return True

    def readinto(self, b):
        while self.i < len(self.fs):
            n = self.fs[self.i].readinto(b)
            if n:
                self.okunan += n
                return n
            self.fs[self.i].close()
            self.i += 1
        return 0


def isle(ham):
    try:
        if ham[:2] == b"\x1f\x8b":
            ham = gzip.decompress(ham)
        o = orjson.loads(ham)
    except Exception:  # bozuk tek iz dosyası (ör. 29.01.2026 "invalid block type"): o uçağı atla, günü kaybetme
        return None
    tr = o.get("trace")
    if not tr:
        return None
    n = len(tr)
    lat = np.fromiter((p[1] for p in tr), float, n)
    lon = np.fromiter((p[2] for p in tr), float, n)
    isabet = []
    for apt, (la, lo, el) in AP.items():
        m = (np.abs(lat - la) < DLAT) & (np.abs(lon - lo) < DLON[apt])
        if m.any():
            isabet.append((apt, m, el))
    if not isabet:
        return None
    irt = [p[3] for p in tr]
    yer = np.fromiter((a == "ground" for a in irt), bool, n)
    alt = np.fromiter(((a if isinstance(a, (int, float)) else np.nan) for a in irt), float, n)
    parcalar = []
    cs_son = None
    cs = None
    for apt, m, el in isabet:
        keep = m & (yer | (alt <= el + UST))
        if not keep.any():
            continue
        if cs is None:
            cs = np.empty(n, dtype=object)
            for i, p in enumerate(tr):
                if len(p) > 8 and isinstance(p[8], dict):
                    f = p[8].get("flight")
                    if f:
                        cs_son = f.strip()
                cs[i] = cs_son
        idx = np.nonzero(keep)[0]
        t0 = float(o["timestamp"])
        parcalar.append(pd.DataFrame({
            "hex": o.get("icao"), "reg": o.get("r"), "tip": o.get("t"), "apt": apt,
            "t": t0 + np.fromiter((tr[i][0] for i in idx), float, len(idx)),
            "lat": lat[idx].astype(np.float32), "lon": lon[idx].astype(np.float32),
            "alt": alt[idx].astype(np.float32), "yer": yer[idx],
            "gs": np.array([tr[i][4] if tr[i][4] is not None else np.nan for i in idx], dtype=np.float32),
            "trk": np.array([tr[i][5] if tr[i][5] is not None else np.nan for i in idx], dtype=np.float32),
            "bayrak": np.array([tr[i][6] or 0 for i in idx], dtype=np.int16),
            "kaynak": [tr[i][9] if len(tr[i]) > 9 else None for i in idx],
            "cs": cs[idx]}))
    return pd.concat(parcalar, ignore_index=True) if parcalar else None


def gun_isle(g, havuz):
    # 2 GB altındaki günler bölünmeden tek .tar olarak yayımlanıyor (ör. 11/18/19/21.01.2025)
    parts = sorted((HAM / g).glob("*.tar.a?")) or sorted((HAM / g).glob("*.tar"))
    if not (HAM / g / ".tamam").exists() or not parts:
        print(f"{g}: arşiv yok ya da eksik", flush=True)
        return
    t0 = time.time()
    zincir = Zincir(parts)
    akis = io.BufferedReader(zincir, buffer_size=16 << 20)
    sonuc, parti, n_iz, n_tut = [], [], 0, 0
    with tarfile.open(fileobj=akis, mode="r|") as tar:
        for mem in tar:
            if not mem.isfile() or "trace_full_" not in mem.name:
                continue
            parti.append(tar.extractfile(mem).read())
            if len(parti) >= PARTI:
                r = [x for x in havuz.map(isle, parti, chunksize=25) if x is not None]
                n_iz += len(parti)
                n_tut += len(r)
                sonuc += r
                parti = []
        if parti:
            r = [x for x in havuz.map(isle, parti, chunksize=25) if x is not None]
            n_iz += len(parti)
            n_tut += len(r)
            sonuc += r
    if zincir.okunan < zincir.toplam - (64 << 20):
        # 29.01.2026 staging: arşiv içi bozukluk tar akışını sessizce erken bitirdi (33 bin iz, diğer günler 46–69 bin)
        raise RuntimeError(f"tar akışı erken bitti: {zincir.okunan / 1e9:.2f} / {zincir.toplam / 1e9:.2f} GB okundu")
    df = pd.concat(sonuc, ignore_index=True)
    for c in ("hex", "reg", "tip", "apt", "kaynak", "cs"):
        df[c] = df[c].astype("category")
    NOKTA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(NOKTA / f"{g}.parquet", index=False)
    print(f"{g}: iz {n_iz:,} | tutulan uçak-havalimanı {n_tut:,} | nokta {len(df):,} | "
          f"havalimanı başına nokta {df['apt'].value_counts().to_dict()} | {time.time() - t0:.0f} sn", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    gunler = [a for a in sys.argv[1:] if "=" not in a]
    with Pool(int(opt.get("isci", 10))) as havuz:
        for g in gunler:
            gun_isle(dt.date.fromisoformat(g).isoformat(), havuz)

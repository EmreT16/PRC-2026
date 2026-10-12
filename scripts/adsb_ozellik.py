"""ADS-B noktalarından (adsb_cikar.py) kalkış başına özellik: izden görülen hareket başlangıcı ve ilk görülme.

Eşleştirme (kalkış satırı ↔ uçak):
  1) CALLSIGN_flt ile aynı çağrı kodunu taşıyan ve kalkışı MVT_TIME'a ±5 dk yakın olan uçak (hex);
  2) yoksa aynı havalimanında MVT_TIME'a ±90 sn içinde yerden havaya geçişi olan TEK uçak.
Kalkış: iz "yer / alçak" durumundan "havada (rakım + 300 ft üstü)" durumuna geçtiği an.
Taksi kesiti: kalkıştan geriye, ardışık boşluk ≤ 600 sn ve havada nokta yokken uzanan noktalar. Kesitte ≥ 600 sn süren
durağanlık (park, transponder açık) varsa kesit onun bitişinden başlar.
Özellikler (sn): ADSB_ILK = MVT − ilk nokta, ADSB_HAREKET = MVT − ilk hareket (hız ≥ 2 kt ya da ilk noktadan > 25 m),
ADSB_TO_FARK = izdeki ilk havada nokta − MVT, ADSB_N, ADSB_BOSLUK (kesitteki en büyük boşluk), ADSB_ILK_GS, ADSB_PARK,
ADSB_ESLESME (1 çağrı kodu, 2 zaman).
Kullanım: python scripts/adsb_ozellik.py 2025-07-15 [2026-07-14 ...]   → data/adsb/oz_<gün>.parquet (+ eğitim günlerinde özet)
"""
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

NOKTA = Path(os.environ.get("PRC_ADSB_KOK", str(Path(__file__).resolve().parents[2] / "PRC_adsb"))) / "noktalar"
# 30.09: PRC_ADSB_OZ = özellik klasörü (varsayılan data/adsb); PRC_ADSB_GEVSEK=1 → üçüncü eşleştirme (boşluklu iz):
# kalkış anı tespit edilemeyen (pistte kapsama boşluğu) ama aynı çağrı koduyla yerde görülen izler, bilinen kalkış
# saatine göre eşleştirilir (ADSB_ESLESME = 3). Ocak 2026 EGLL/EDDF eşleşmeyenlerinin çoğu böyle.
CIKTI = config.DATA_DIR / os.environ.get("PRC_ADSB_OZ", "adsb")
GEVSEK = os.environ.get("PRC_ADSB_GEVSEK") == "1"
# 01.10: PRC_ADSB_IZ5=1 → taksi başı ve park ayracı.
# 2025 EGLL izleri seyrek, 2026 yoğun: itme 2 kt altında başlıyor,
# park yerinde konum titremesi "> 25 m" koşulunu erken tetikliyor. Park merkezi data/park_merkez.parquet (adsb4'ten).
IZ5 = os.environ.get("PRC_ADSB_IZ5") == "1"
# 01.10 akşam: PRC_ADSB_SEYRELT=p → yerdeki noktaların p kadarı tutulur (havadakiler korunur; gün tohumlu), yalnız 2025
# günleri. Düzelticiyi seyrek izlere dayanıklı kılmak için eğitim kopyaları (yoğun → seyrek
# kazancı düşüyor; üç yoğunlukla birlikte eğitilen model her yoğunlukta en iyi). Çıktı klasörü PRC_ADSB_OZ ile seçilir.
SEYRELT = float(os.environ.get("PRC_ADSB_SEYRELT", "1"))
R_PARK = 50.0
_PARK_MERKEZ = None
RAW = config.PRC2_DIR / "data"
ELEV = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12, "LEMD": 2000, "LFPG": 392, "LIRF": 13,
        "LSZH": 1416, "LTFM": 325}
BOSLUK, PARK, HAVA = 600.0, 600.0, 300.0
KOLON = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "CALLSIGN_flt", "FLIGHT_mvt",
         "AIRCRAFT_TYPE_mvt", "TAXITIME_SEC_mvt", "STAND_mvt"]


def epoch(s):
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)


def kalkislar(g):
    gun = dt.date.fromisoformat(g)
    f = config.RANK_RAW if gun.year == 2026 else \
        RAW / f"training_{gun:%Y-%m}-01_{(gun.replace(day=28) + dt.timedelta(days=4)).replace(day=1):%Y-%m-%d}.parquet"
    df = pd.read_parquet(f, columns=KOLON)
    df = df[df["PHASE_mvt"].eq("DEP") & df["MVT_TIME_UTC_mvt"].dt.date.eq(gun)].reset_index(drop=True)
    df["mvt"] = epoch(df["MVT_TIME_UTC_mvt"])
    return df


def mesafe_m(lat1, lon1, lat2, lon2):
    k = np.cos(np.radians(lat1))
    return 111_195.0 * np.hypot(lat2 - lat1, (lon2 - lon1) * k)


def ucak_dizileri(p):
    if p.empty:
        return {}
    p = p.sort_values(["hex", "t"], kind="stable")
    hx = p["hex"].astype(str).to_numpy()
    sinir = np.flatnonzero(np.r_[True, hx[1:] != hx[:-1], True])
    kol = {c: p[c].to_numpy() for c in ("t", "lat", "lon", "alt", "yer", "gs")}
    kol["cs"] = p["cs"].astype(object).to_numpy()
    out = {}
    for a, b in zip(sinir[:-1], sinir[1:]):
        out[hx[a]] = {c: v[a:b] for c, v in kol.items()}
    return out


def kalkis_anlari(u, el):
    hava = (~u["yer"]) & (u["alt"] > el + HAVA)
    alcak = u["yer"] | ((u["alt"] <= el + HAVA) & (np.nan_to_num(u["gs"], nan=0.0) < 200))
    i = np.flatnonzero(hava[1:] & alcak[:-1] & (np.diff(u["t"]) <= 180)) + 1
    return i  # ilk havada noktanın indisi


def kesit(u, i_hava, mvt, plat=np.nan, plon=np.nan):
    t, lat, lon, gs = u["t"], u["lat"], u["lon"], u["gs"]
    son = i_hava - 1
    j = son
    varis = False
    while j > 0 and t[j] - t[j - 1] <= BOSLUK:
        if u["alt"][j - 1] > 0 and not u["yer"][j - 1] and u["alt"][j - 1] > u["alt"][son] + HAVA:
            varis = True  # kesit, aynı transponder oturumundaki önceki inişe dayandı
            break
        j -= 1
    idx = np.arange(j, son + 1)
    g = np.nan_to_num(gs[idx], nan=0.0)
    d = np.r_[0.0, mesafe_m(lat[idx[:-1]], lon[idx[:-1]], lat[idx[1:]], lon[idx[1:]])]
    duran = (g < 2) & (d < 25)
    park = False
    bas = 0
    if varis:
        # iniş → park (transponder açık) → itme: inişten sonraki İLK uzun durağan koşu park yeridir. Sonraki uzun
        # duruşlar (buz çözme, kuyruk) park sayılmaz; eski "son uzun koşu" kuralı buz çözmeyi park sanabiliyordu.
        k = 0
        while k < len(idx) - 1:
            if duran[k]:
                e = k
                while e + 1 < len(idx) and duran[e + 1]:
                    e += 1
                if t[idx[e]] - t[idx[k]] >= PARK:
                    park, bas = True, e
                    break
                k = e + 1
            else:
                k += 1
    ks = idx[bas:]
    gk = np.nan_to_num(gs[ks], nan=0.0)
    d0 = mesafe_m(lat[ks[0]], lon[ks[0]], lat[ks], lon[ks])
    hareket = np.flatnonzero((gk >= 2) | (d0 > 25))
    # kalkış koşusu başlangıcı: sondan geriye hızın ilk kez 40 kt altına indiği nokta
    kos = np.flatnonzero(gk < 40)
    i_kos = ks[kos[-1]] if len(kos) else ks[-1]
    t_har = t[ks[hareket[0]]] if len(hareket) else np.nan
    if np.isfinite(t_har) and t_har > mvt - 60:
        t_har = np.nan  # hareket kalkışa 1 dk'dan yakınsa iz seyrek: geçersiz
    ilk_dur = t[ks[hareket[0]]] - t[ks[0]] if len(hareket) else 0.0
    # hareket başlangıcından koşuya: yol uzunluğu ve toplam durma süresi
    if np.isfinite(t_har):
        a = ks[hareket[0]]
        seg = np.arange(a, i_kos + 1)
        adim = mesafe_m(lat[seg[:-1]], lon[seg[:-1]], lat[seg[1:]], lon[seg[1:]])
        dt_ = np.diff(t[seg])
        dur = (np.nan_to_num(gs[seg[1:]], nan=0.0) < 2) & (adim < 25)
        yol, durma = float(adim.sum()), float(dt_[dur].sum())
        bosluk5 = float(dt_[t[seg[1:]] <= t_har + 300].max()) if (t[seg[1:]] <= t_har + 300).any() else np.nan
    else:
        yol = durma = bosluk5 = np.nan
    i_har = ks[hareket[0]] if len(hareket) else ks[0]
    ek = iz5(t, lat, lon, gs, ks, mvt, plat, plon) if IZ5 else {}
    return {**ek, "ADSB_ILK": mvt - t[ks[0]], "ADSB_HAREKET": mvt - t_har, "ADSB_TO_FARK": t[i_hava] - mvt,
            "ADSB_N": len(ks), "ADSB_BOSLUK": float(np.max(np.diff(t[ks]))) if len(ks) > 1 else np.nan,
            "ADSB_BOSLUK5": bosluk5, "ADSB_ILK_GS": float(gs[ks[0]]), "ADSB_PARK": park,
            "ADSB_ILK_DURMA": float(ilk_dur),
            "ADSB_ILK_UZAK": float(mesafe_m(lat[ks[0]], lon[ks[0]], lat[i_kos], lon[i_kos])),
            "ADSB_YOL": yol, "ADSB_DURMA": durma, "ADSB_KOSU": mvt - t[i_kos],
            # konumlar (30.09): park yeri koordinatlarını veriden çıkarmak ve "park yerinden ilk görülen noktaya uzaklık"
            # için (yolda ilk görülen uçuşlarda görülmeyen taksi kısmı)
            "ADSB_ILK_LAT": float(lat[ks[0]]), "ADSB_ILK_LON": float(lon[ks[0]]),
            "ADSB_HAR_LAT": float(lat[i_har]), "ADSB_HAR_LON": float(lon[i_har]),
            "ADSB_KOSU_LAT": float(lat[i_kos]), "ADSB_KOSU_LON": float(lon[i_kos])}


def iz5(t, lat, lon, gs, ks, mvt, plat, plon):
    """Taksi başı (art arda iki nokta ≥ 4 kt) ve park ayracı: çapaya (park merkezi, yoksa taksiden önceki ilk durağan
    nokta) ≤ 50 m ve durağan (gs < 1 kt; hız yoksa adım < 10 m) olan taksiden önceki SON nokta (P_L) ile sonraki nokta
    (P_U); itme bu aralıkta başlar. P_ILK: park yerindeki ilk durağan nokta. Özellik = MVT − olay anı (sn)."""
    g = gs[ks].astype(float)
    g0 = np.nan_to_num(g, nan=0.0)
    la, lo, tt = lat[ks], lon[ks], t[ks]
    k4 = np.flatnonzero((g0[:-1] >= 4) & (g0[1:] >= 4))
    i4 = k4[0] if len(k4) else len(ks)
    o = {"ADSB_TAKSI_BAS": mvt - tt[k4[0]] if len(k4) else np.nan, "ADSB_PARK_L": np.nan, "ADSB_PARK_U": np.nan,
         "ADSB_PARK_ARALIK": np.nan, "ADSB_PARK_ILK": np.nan}
    adim = np.r_[0.0, mesafe_m(la[:-1], lo[:-1], la[1:], lo[1:])] if len(ks) > 1 else np.zeros(len(ks))
    dur = np.where(np.isnan(g), adim < 10, g < 1)
    if np.isfinite(plat):
        alat, alon = plat, plon
    else:
        j = np.flatnonzero(dur[:i4])
        alat, alon = (la[j[0]], lo[j[0]]) if len(j) else (np.nan, np.nan)
    if np.isfinite(alat):
        j = np.flatnonzero(dur[:i4] & (mesafe_m(alat, alon, la[:i4], lo[:i4]) <= R_PARK))
        if len(j) and j[-1] + 1 < len(ks):
            o.update(ADSB_PARK_L=mvt - tt[j[-1]], ADSB_PARK_U=mvt - tt[j[-1] + 1],
                     ADSB_PARK_ARALIK=tt[j[-1] + 1] - tt[j[-1]], ADSB_PARK_ILK=mvt - tt[j[0]])
    return o


def park_merkezi(apt, park):
    global _PARK_MERKEZ
    if _PARK_MERKEZ is None:
        m = pd.read_parquet(config.DATA_DIR / "park_merkez.parquet")
        _PARK_MERKEZ = {(a, s): (la, lo) for a, s, la, lo in m[["ADEP_mvt", "STAND_mvt", "park_lat", "park_lon"]].itertuples(index=False)}
    return _PARK_MERKEZ.get((apt, park), (np.nan, np.nan))


_IATA = None


def cagri_kodu(r):
    """NM çağrı kodu; yoksa (NM'siz satırlar) hareket verisindeki uçuş numarası, IATA önekiyse ICAO'ya çevrilmiş
    (data/iata_icao.json: NM'li satırlarda uçuş numarası ↔ çağrı kodu çiftlerinden, 30.09). GEVSEK kapalıysa yalnız NM."""
    global _IATA
    if isinstance(r.CALLSIGN_flt, str) and r.CALLSIGN_flt.strip():
        return r.CALLSIGN_flt.strip()
    if not GEVSEK or not isinstance(r.FLIGHT_mvt, str):
        return None
    if _IATA is None:
        import json
        yol = config.DATA_DIR / "iata_icao.json"
        _IATA = json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else {}
    import re
    m = re.match(r"^([A-Z]{2,3}|[A-Z]\d|\d[A-Z])(\d+[A-Z]*)$", r.FLIGHT_mvt.strip())
    if not m:
        return None
    on, no = m.group(1), m.group(2)
    if len(on) == 3:
        return on + no
    return (_IATA[on] + no) if on in _IATA else None


def bosluklu(ucak, cs_hex, cs, mvt, el):
    """Üçüncü eşleştirme: çağrı kodu cs olan izlerden, kalkıştan önceki 1 saatte yerde / alçakta noktası olan ve
    ardından en az bir noktası bulunan iz. (hex, i_hava) döner; i_hava = son yer noktasından sonraki ilk nokta."""
    en = None
    for hx in cs_hex.get(cs, ()):
        u = ucak[hx]
        t = u["t"]
        alcak = u["yer"] | ((u["alt"] <= el + HAVA) & (np.nan_to_num(u["gs"], nan=0.0) < 200))
        k = np.flatnonzero(alcak & (t < mvt + 60) & (t >= mvt - 3600))
        if len(k) == 0 or k[-1] + 1 >= len(t) or t[k[-1] + 1] > mvt + 900:
            continue
        aday = (hx, int(k[-1] + 1), mvt - t[k[-1]])
        if en is None or aday[2] < en[2]:
            en = aday
    return None if en is None else en[:2]


def gun(g):
    t0 = time.time()
    dep = kalkislar(g)
    kol = ["hex", "apt", "t", "lat", "lon", "alt", "yer", "gs", "cs"]
    pts = pd.read_parquet(NOKTA / f"{g}.parquet", columns=kol)
    once = NOKTA / f"{dt.date.fromisoformat(g) - dt.timedelta(days=1)}.parquet"
    if once.exists():  # gece yarısından hemen sonraki kalkışlar için önceki günün son 3 saati
        p0 = pd.read_parquet(once, columns=kol)
        p0 = p0[p0["t"] >= pts["t"].min() - 3 * 3600]
        pts = pd.concat([p0, pts], ignore_index=True)
        for c in ("hex", "apt", "cs"):
            pts[c] = pts[c].astype(str).replace({"None": None, "nan": None})
    if SEYRELT < 1:
        yer = pts["yer"].fillna(False).to_numpy(bool)
        tut = ~yer | (np.random.default_rng(int(g.replace("-", ""))).random(len(pts)) < SEYRELT)
        pts = pts[tut].reset_index(drop=True)
    rows = []
    for apt, d_apt in dep.groupby("ADEP_mvt"):
        el = ELEV[apt]
        ucak = ucak_dizileri(pts[pts["apt"].eq(apt)])
        olay = []  # (t_kalkış, hex, indis, çağrı kodu)
        for hx, u in ucak.items():
            for i in kalkis_anlari(u, el):
                cs = next((c for c in u["cs"][max(0, i - 50):i + 5][::-1] if c), None)
                olay.append((u["t"][i], hx, i, cs))
        ev = pd.DataFrame(olay, columns=["t", "hex", "i", "cs"])
        cs_hex = {}
        if GEVSEK:
            for hx, u in ucak.items():
                for c in {str(c).strip() for c in u["cs"] if c}:
                    cs_hex.setdefault(c, []).append(hx)
        kullanilan = set()
        for r in d_apt.itertuples(index=False):
            sec, tur = None, 0
            cs_r = cagri_kodu(r)
            if cs_r and len(ev):
                c = ev[ev["cs"].eq(cs_r) & (ev["t"].sub(r.mvt).abs() <= 300)]
                if len(c):
                    sec, tur = c.iloc[(c["t"] - r.mvt).abs().argmin()], 1
            if sec is None and len(ev):
                c = ev[(ev["t"].sub(r.mvt).abs() <= 90) & ~ev.index.isin(kullanilan)]
                if len(c) == 1:
                    sec, tur = c.iloc[0], 2
            oz = {"MVT_ID_mvt": r.MVT_ID_mvt, "ADSB_ESLESME": tur}
            if sec is not None:
                kullanilan.add(sec.name)
                oz.update(kesit(ucak[sec["hex"]], int(sec["i"]), r.mvt, *(park_merkezi(apt, r.STAND_mvt) if IZ5 else ())))
            elif GEVSEK and cs_r:
                b3 = bosluklu(ucak, cs_hex, cs_r, r.mvt, el)
                if b3 is not None:
                    oz["ADSB_ESLESME"] = 3
                    oz.update(kesit(ucak[b3[0]], b3[1], r.mvt, *(park_merkezi(apt, r.STAND_mvt) if IZ5 else ())))
            rows.append(oz)
    oz = pd.DataFrame(rows)
    CIKTI.mkdir(parents=True, exist_ok=True)
    oz.to_parquet(CIKTI / f"oz_{g}.parquet", index=False)
    m = dep.merge(oz, on="MVT_ID_mvt", how="left")
    print(f"{g}: kalkış {len(dep):,} | eşleşen {int((m['ADSB_ESLESME'] > 0).sum()):,} "
          f"(çağrı kodu {int((m['ADSB_ESLESME'] == 1).sum()):,}) | {time.time() - t0:.0f} sn", flush=True)
    return m


def ozet(m):
    y = m["TAXITIME_SEC_mvt"].to_numpy(float)
    nm = m["mvt"].to_numpy() - epoch(m["AOBT_3_flt"])
    rows = {}
    for apt, i in m.groupby("ADEP_mvt").indices.items():
        a = m.iloc[i]
        e = a["ADSB_ESLESME"].fillna(0).to_numpy() > 0
        r = {"kalkış": len(a), "eşleşen %": 100 * e.mean(),
             "TO farkı medyan": a.loc[e, "ADSB_TO_FARK"].median() if e.any() else np.nan}
        if np.isfinite(y[i]).any():
            ok = e & np.isfinite(y[i]) & (y[i] < 7200)
            for ad, v in (("HAREKET", a["ADSB_HAREKET"].to_numpy(float)), ("ILK", a["ADSB_ILK"].to_numpy(float)),
                          ("NM", nm[i])):
                h = v[ok] - y[i][ok]
                h = h[np.isfinite(h)]
                r[f"{ad} medyan hata"] = np.median(h) if len(h) else np.nan
                r[f"{ad} |hata|≤60 %"] = 100 * np.mean(np.abs(h) <= 60) if len(h) else np.nan
                r[f"{ad} RMSE"] = np.sqrt(np.mean(h ** 2)) if len(h) else np.nan
        rows[apt] = r
    print(pd.DataFrame(rows).T.round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    args = sys.argv[1:]
    if args and args[0] == "hepsi":  # noktası olan Ocak/Temmuz günleri: özelliği yoksa ya da eskimişse
        yeniden = "yeniden" in args

        def eski(p):
            oz = CIKTI / f"oz_{p.stem}.parquet"
            if yeniden or not oz.exists():
                return True
            import pyarrow.parquet as pq
            if "ADSB_ILK_LAT" not in pq.read_schema(oz).names:  # konum sütunları öncesi (30.09) üretilmiş
                return True
            once = NOKTA / f"{dt.date.fromisoformat(p.stem) - dt.timedelta(days=1)}.parquet"
            # kendi noktaları ya da (gece yarısı uçuşları için) önceki günün noktaları özellikten yeniyse
            return p.stat().st_mtime > oz.stat().st_mtime or (once.exists() and once.stat().st_mtime > oz.stat().st_mtime)

        aylar = {"2025": ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"), "2026": tuple(f"{m:02d}" for m in config.TEST_AYLAR)}  # etiketli / test ayları (02.10: + Mart, Kasım; 08.10: test ayları config.TEST_AYLAR, finalde 1, 2, 6, 7)
        if SEYRELT < 1:  # seyreltilmiş kopyalar yalnız eğitim için
            aylar.pop("2026")
        args = [p.stem for p in sorted(NOKTA.glob("20*.parquet"))
                if p.stem[5:7] in aylar.get(p.stem[:4], ()) and eski(p)]
        isci = int(next((a.split("=")[1] for a in sys.argv if a.startswith("isci=")), "1"))
        if isci > 1:  # günler bağımsız: paralel
            from multiprocessing import Pool
            with Pool(isci) as havuz:
                havuz.map(gun, args, chunksize=1)
        else:
            for g in args:
                gun(g)
    else:
        for g in args:
            ozet(gun(dt.date.fromisoformat(g).isoformat()))

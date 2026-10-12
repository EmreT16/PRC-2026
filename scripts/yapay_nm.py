"""Yapay NM off-block yığınları: havalimanı başına tespit ve doğrulama (yalnız 2025 eğitim kalkışları).

NM taksi = MVT − AOBT_3 (sn). Yapay AOBT_3 = kalkış − sabit (havalimanı/pist varsayılan taksisi) ise NM taksi dar bir
kutuda yığılır. Tespit (MVT saniyesi ≠ 0 olan kalkışlar):
  2 sn kutular; taban = [b−180, b−40] ∪ [b+40, b+180] aralığındaki kutu sayılarının medyanı;
  tepe kutusu: sayı ≥ 5 × taban ve sayı ≥ 40; bitişik tepe kutuları (≤ 6 sn aralık) tek küme.
Not: EDDM, EGLL, LFPG, LIRF, LTFM'de her tam dakikada (±6 sn) zaman damgası çözünürlüğünden gelen yığınlar da çıkar;
bunlar yapay DEĞİL. Karar kuralı main() içinde (üç koşul, hepsi 2025 verisiyle).
Çıktı: kutuk/yapay_nm_kumeleri.csv (küme listesi + doğrulama sütunları); scripts/wx_ozellik.py karar == "yapay"
olan kümeleri okur (NM_YAPAY, NM_YAPAY_MERKEZ).
Kullanım: python scripts/yapay_nm.py [çıktı=kutuk/yapay_nm_kumeleri.csv]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

VERI = config.ROOT.parent / "data"
KOL = ["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "RUNWAY_mvt"]


def naive(s):
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if s.dt.tz is not None else s


def oku():
    fs = sorted(glob.glob(str(VERI / "training_2025-*.parquet")))
    assert len(fs) == 12, f"{VERI}: 12 eğitim dosyası beklenir, {len(fs)} bulundu"
    d = pd.concat([pq.read_table(f, columns=KOL, filters=[("PHASE_mvt", "=", "DEP")]).to_pandas() for f in fs],
                  ignore_index=True)
    m, a = naive(d.MVT_TIME_UTC_mvt), naive(d.AOBT_3_flt)
    d["nm"] = (m - a).dt.total_seconds()
    d["ofs"] = (naive(d.BLOCK_TIME_UTC_mvt) - a).dt.total_seconds()
    d["ay"] = m.dt.strftime("%Y-%m")
    d["mvt_sn"] = m.dt.second
    hd = pd.to_numeric(d.RUNWAY_mvt.astype(str).str.extract(r"^(\d{2})")[0], errors="coerce") * 10
    d["yon"] = np.where(hd.isna(), "?", np.where((hd > 45) & (hd <= 225), "D-G", "B-K"))  # doğu/güney ↔ batı/kuzey
    return d[d.nm.between(0, 3600)]


def kumeler(x):
    b = (x // 2).astype(int)
    say = np.bincount(b, minlength=1801).astype(float)
    taban = np.zeros_like(say)
    for i in range(len(say)):
        idx = list(range(max(0, i - 90), max(0, i - 20))) + list(range(min(len(say), i + 20), min(len(say), i + 90)))
        taban[i] = np.median(say[idx]) if idx else 0
    tepe = np.where((say >= 5 * np.maximum(taban, 1)) & (say >= 40))[0]
    out, grup = [], []
    for t in tepe:
        if grup and t - grup[-1] > 3:
            out.append(grup); grup = []
        grup.append(t)
    if grup:
        out.append(grup)
    sonuc = []
    for g in out:
        lo, hi = g[0] * 2, g[-1] * 2 + 2
        w = say[g] - taban[g]
        sonuc.append(dict(alt=lo, ust=hi, merkez=round(float(np.average(np.array(g) * 2 + 1, weights=w)), 1),
                          fazla=int(w.sum())))
    return sonuc


def main(cikti):
    tr = oku()
    satirlar = []
    for apt, g in tr.groupby("ADEP_mvt"):
        ks = kumeler(g.loc[g.mvt_sn != 0, "nm"].to_numpy())
        herhangi = np.zeros(len(g), bool)          # herhangi bir aday kümenin içinde mi
        for k in ks:
            herhangi |= g.nm.between(k["alt"], k["ust"], inclusive="left").to_numpy()
        for k in ks:
            ic = g.nm.between(k["alt"], k["ust"], inclusive="left").to_numpy()
            e = g[ic]
            # komşular: aynı NM aralığında (±180 sn; yetmezse ±600 sn) hiçbir aday kümeye düşmeyen satırlar
            for w in (180, 600):
                km = (~herhangi) & g.nm.between(k["merkez"] - w, k["merkez"] + w).to_numpy()
                if km.sum() >= 50:
                    break
            kd = g[km]
            k.update(ICAO=apt, n_2025=len(e), pay_2025=round(len(e) / len(g) * 100, 2),
                     yon_BK=round((e.yon == "B-K").mean() * 100, 1),
                     blok_60_ic=round((e.ofs.abs() <= 60).mean() * 100, 1),
                     blok_60_komsu=round((kd.ofs.abs() <= 60).mean() * 100, 1) if len(kd) else np.nan,
                     hedef_med_ic=e.TAXITIME_SEC_mvt.median(),
                     hedef_med_komsu=kd.TAXITIME_SEC_mvt.median() if len(kd) else np.nan,
                     komsu_n=int(km.sum()), komsu_pencere=w)
            satirlar.append(k)
    K = pd.DataFrame(satirlar)[["ICAO", "merkez", "alt", "ust", "fazla", "n_2025", "pay_2025", "yon_BK",
                               "blok_60_ic", "blok_60_komsu", "hedef_med_ic", "hedef_med_komsu", "komsu_n",
                               "komsu_pencere"]]
    # Not: EDDM/EGLL/LFPG/LIRF/LTFM'de NM taksi ~%100 tam dakika; "komşu" satırlar orada çok az (komsu_n) ve anlamsız.
    # Karar üç koşulla (hepsi):
    #  (1) hedef NM değerini izlemiyor: |hedef_med_ic − merkez| ≥ 150 sn
    #  (2) BLOCK–AOBT uyumu düşük: blok_60_ic ≤ 0,5 × ref; ref = aynı havalimanında |hedef − merkez| < 100 olan
    #      (açıkça gerçek) kümelerin blok_60 medyanı (yoksa havalimanı geneli)
    #  (3) tekil sıçrama: n ≥ 2 × min(n[merkez − 60], n[merkez + 60]) (±7 sn pencerelerde)
    K["hedef_eksi_merkez"] = (K.hedef_med_ic - K.merkez).round(0)
    ref, sicrama = {}, []
    for apt, k in K.groupby("ICAO"):
        g = tr[tr.ADEP_mvt == apt]
        gercek = k[k.hedef_eksi_merkez.abs() < 100]
        ref[apt] = gercek.blok_60_ic.median() if len(gercek) else (g.ofs.abs() <= 60).mean() * 100
        for r in k.itertuples():
            nk = [int(g.nm.between(r.merkez + o - 7, r.merkez + o + 7).sum()) for o in (-60, 60)]
            sicrama.append((r.Index, round(r.n_2025 / max(min(nk), 1), 1)))
    K["blok_ref"] = K.ICAO.map(ref).round(1)
    K["sicrama_orani"] = pd.Series(dict(sicrama))
    K["karar"] = np.where((K.hedef_eksi_merkez.abs() >= 150) & (K.blok_60_ic <= 0.5 * K.blok_ref)
                          & (K.sicrama_orani >= 2), "yapay", "gercek/zaman damgasi")
    K.to_csv(cikti, index=False)
    pd.set_option("display.width", 250)
    print(K[K.karar == "yapay"].to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else config.LOG_DIR / "yapay_nm_kumeleri.csv")

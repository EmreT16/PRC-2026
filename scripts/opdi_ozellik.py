"""OPDI yer olaylarından (scripts/opdi_indir.py) kalkış başına özellik → data/opdi_oz.parquet.

07.10: EDDF 2026'da adsb.lol kalkışların %32'sini yalnız kalkış koşusunda görüyor;
OPDI bunların ~%74'ünde uçağı yerde görüyor (LSZH'de de zengin). Diğer havalimanlarında OPDI'nin yer olayı çok az.

Eşleştirme (kalkış ↔ OPDI flight_id), olaylar kalkıştan (MVT) 3 sa önce ile 5 dk sonra arasında:
  1) aynı havalimanı + çağrı kodu (CALLSIGN_flt = OPDI osn_flight_id). Birden çok aday varsa son pist girişi MVT'ye en
     yakın olan (pist olayı yoksa son olayı en geç olan);
  2) eşleşmeyenler: aynı havalimanında pistten çıkışı (exit-runway) MVT − 30 … MVT + 60 sn içinde olan TEK uçuş
     (pist biliniyorsa aynı pist), çağrı koduyla başka kalkışa eşlenmemiş.
Özellikler (sn, MVT'den önce pozitif):
  OP_ILK (ilk olay), OP_YER (ilk yer olayı: park / apron / taxiway), OP_PARK (en uzun kalınan park yerinden çıkış),
  OP_PARK_SURE (orada geçen süre), OP_PARK_SON (son park poligonu çıkışı), OP_APRON (son apron çıkışı),
  OP_TW (ilk taxiway girişi), OP_PIST (son pist girişi), OP_PIST_SURE, OP_TW_TOPLAM / OP_TW_ENUZUN (taxiway
  poligonlarında süre), OP_TW_SAY, OP_N, OP_PARK_UYUM / OP_PIST_UYUM (OSM adı kayıttaki park yeri / pistle aynı mı),
  OP_ESLESME (1 çağrı kodu, 2 pist zamanı).
Kullanım: python scripts/opdi_ozellik.py   (bütün eğitim + test kalkışları; ay ay)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

SUZ = Path(os.environ.get("PRC_ADSB_KOK", str(Path(__file__).resolve().parents[2] / "PRC_adsb"))) / "opdi" / "suz"
ONCE, SONRA = 3 * 3600.0, 300.0
YER = {"exit-parking_position", "entry-parking_position", "entry-apron", "exit-apron", "entry-taxiway", "exit-taxiway"}
CIKTI = config.DATA_DIR / "opdi_oz.parquet"


def kalkislar():
    A.EPOCH_TAM = True
    k = ["MVT_ID_mvt", "ADEP_mvt", "CALLSIGN_flt", "STAND_mvt", "RUNWAY_mvt"]
    d = pd.concat([pd.read_parquet(f, columns=k) for f in (config.DEP_TRAIN, config.DEP_RANK)], ignore_index=True)
    d = d.drop_duplicates("MVT_ID_mvt").merge(A.mvt_epoch(set(d["MVT_ID_mvt"])), on="MVT_ID_mvt", how="inner")
    d["CALLSIGN_flt"] = d["CALLSIGN_flt"].where(d["CALLSIGN_flt"].notna()).str.strip().str.upper().replace("", np.nan)
    return d


def olaylar(t0, t1):
    fs = []
    for f in sorted(SUZ.glob("fe_*.parquet")):
        a, b = (pd.Timestamp(x, tz="UTC").timestamp() for x in f.stem[3:].split("_"))
        if b >= t0 - ONCE and a <= t1 + SONRA:
            fs.append(pd.read_parquet(f))
    if not fs:
        return None
    o = pd.concat(fs, ignore_index=True)
    o = o[(o["t"] >= t0 - ONCE) & (o["t"] <= t1 + SONRA)].drop_duplicates(["flight_id", "type", "t", "osm_ref"])
    cs = o.dropna(subset=["cs"]).groupby("flight_id")["cs"].agg(lambda s: s.mode().iloc[0])
    o["csf"] = o["flight_id"].map(cs).str.strip().str.upper().replace("", np.nan)
    return o.reset_index(drop=True)


def ozet(m):
    """m: (MVT_ID_mvt, flight_id) çiftleri × olaylar (pencere içinde). Çift başına özellik."""
    m = m.sort_values("t")
    m["once"] = m["mvt"] - m["t"]
    g = m.groupby(["MVT_ID_mvt", "flight_id"], sort=False)
    o = g.agg(OP_ILK=("once", "max"), OP_SON=("once", "min"), OP_N=("once", "size")).reset_index()

    def tip_ag(tip, ad, f, kol="once"):
        x = m[m["type"] == tip].groupby(["MVT_ID_mvt", "flight_id"])[kol].agg(f).rename(ad)
        return x

    parts = [
        m[m["type"].isin(YER)].groupby(["MVT_ID_mvt", "flight_id"])["once"].max().rename("OP_YER"),
        tip_ag("exit-parking_position", "OP_PARK_SON", "min"),
        tip_ag("exit-apron", "OP_APRON", "min"),
        tip_ag("entry-taxiway", "OP_TW", "max"),
        tip_ag("entry-taxiway", "OP_TW_SAY", "size"),
        tip_ag("entry-taxiway", "OP_TW_TOPLAM", "sum", "kullanim_s"),
        tip_ag("entry-taxiway", "OP_TW_ENUZUN", "max", "kullanim_s"),
    ]
    # son pist girişi (kalkış pisti): pistte süre ve MVT'ye uzaklık
    p = m[m["type"] == "entry-runway"].drop_duplicates(["MVT_ID_mvt", "flight_id"], keep="last")
    p = p.set_index(["MVT_ID_mvt", "flight_id"])
    parts += [p["once"].rename("OP_PIST"), p["kullanim_s"].rename("OP_PIST_SURE"), p["osm_ref"].rename("pist_ref")]
    # en uzun kalınan park yeri (bekleme yeri): çıkış anı ve süre
    k = m[m["type"] == "exit-parking_position"].sort_values("kullanim_s").drop_duplicates(
        ["MVT_ID_mvt", "flight_id"], keep="last").set_index(["MVT_ID_mvt", "flight_id"])
    parts += [k["once"].rename("OP_PARK"), k["kullanim_s"].rename("OP_PARK_SURE"), k["osm_ref"].rename("park_ref")]
    o = o.set_index(["MVT_ID_mvt", "flight_id"]).join(parts).reset_index()
    return o


def ay_isle(d, o):
    t0 = time.time()
    # 1) çağrı kodu: olaylar, MVT'nin olası günleriyle anahtarlanır (gece yarısını aşan pencere)
    gun = lambda x: (x // 86400).astype(np.int64)  # noqa: E731
    o1 = pd.concat([o.assign(g=gun(o["t"] - SONRA)), o.assign(g=gun(o["t"] + ONCE))]).drop_duplicates(
        ["flight_id", "type", "t", "osm_ref", "g"])
    dk = d[d["CALLSIGN_flt"].notna()].assign(g=gun(d["mvt"]))
    o1 = o1[o1["csf"].notna()]
    m = dk.merge(o1, left_on=["ADEP_mvt", "CALLSIGN_flt", "g"], right_on=["apt", "csf", "g"], how="inner")
    m = m[(m["t"] >= m["mvt"] - ONCE) & (m["t"] <= m["mvt"] + SONRA)]
    s = ozet(m)
    s["skor"] = (s["OP_PIST"] - 30).abs().fillna(1e6 + s["OP_SON"].abs())  # pist olayı yoksa son olayı MVT'ye en yakın
    s = s.sort_values("skor").drop_duplicates("MVT_ID_mvt").assign(OP_ESLESME=1)
    # 2) pist zamanı: eşleşmeyen kalkışlar, tek aday
    kalan = d[~d["MVT_ID_mvt"].isin(s["MVT_ID_mvt"])]
    kul = set(s["flight_id"])
    xr = o[(o["type"] == "exit-runway") & ~o["flight_id"].isin(kul)][["flight_id", "apt", "t", "osm_ref"]]
    km = kalan.assign(g=gun(kalan["mvt"])).merge(
        pd.concat([xr.assign(g=gun(xr["t"] - 60)), xr.assign(g=gun(xr["t"] + 30))]).drop_duplicates(),
        left_on=["ADEP_mvt", "g"], right_on=["apt", "g"], how="inner")
    km = km[(km["t"] >= km["mvt"] - 30) & (km["t"] <= km["mvt"] + 60)]
    ayni = np.array([pd.isna(p) or str(p) in str(r).split("/") for p, r in zip(km["RUNWAY_mvt"], km["osm_ref"])], bool)
    km = km[ayni]
    km = km.drop_duplicates(["MVT_ID_mvt", "flight_id"])
    n = km.groupby("MVT_ID_mvt")["flight_id"].transform("size")
    n2 = km.groupby("flight_id")["MVT_ID_mvt"].transform("size")
    km = km[(n == 1) & (n2 == 1)][["MVT_ID_mvt", "flight_id"]]
    s2 = None
    if len(km):
        m2 = km.merge(d[["MVT_ID_mvt", "mvt", "ADEP_mvt"]], on="MVT_ID_mvt").merge(o, on="flight_id")
        m2 = m2[(m2["apt"] == m2["ADEP_mvt"]) & (m2["t"] >= m2["mvt"] - ONCE) & (m2["t"] <= m2["mvt"] + SONRA)]
        s2 = ozet(m2).assign(OP_ESLESME=2)
    s = pd.concat([s, s2], ignore_index=True) if s2 is not None else s
    s = s.merge(d[["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]], on="MVT_ID_mvt", how="left")
    s["OP_PARK_UYUM"] = np.where(s["park_ref"].isna(), np.nan,
                                 (s["park_ref"].astype(str).str.upper() == s["STAND_mvt"].astype(str).str.upper()))
    s["OP_PIST_UYUM"] = np.where(s["pist_ref"].isna() | s["RUNWAY_mvt"].isna(), np.nan,
                                 [str(r) in str(p).split("/") for r, p in zip(s["RUNWAY_mvt"], s["pist_ref"])])
    print(f"  {len(d):,} kalkış → çağrı kodu {int((s['OP_ESLESME'] == 1).sum()):,}, pist zamanı "
          f"{int((s['OP_ESLESME'] == 2).sum()):,} | yerde görülen (OP_YER ≥ 150) {int((s['OP_YER'] >= 150).sum()):,} "
          f"| {time.time() - t0:.0f} sn", flush=True)
    return s


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = kalkislar()
    d["ay"] = pd.to_datetime(d["mvt"], unit="s").dt.strftime("%Y-%m")
    out = []
    secim = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None  # ör. 2025-01 (deneme; çıktı yazılmaz)
    for ay, dd in d.groupby("ay"):
        if secim and ay not in secim:
            continue
        o = olaylar(dd["mvt"].min(), dd["mvt"].max())
        if o is None:
            print(f"{ay}: OPDI dosyası yok", flush=True)
            continue
        print(f"{ay}: {len(o):,} olay", flush=True)
        out.append(ay_isle(dd, o))
    s = pd.concat(out, ignore_index=True)
    kol = ["MVT_ID_mvt", "OP_ESLESME", "OP_ILK", "OP_SON", "OP_YER", "OP_PARK", "OP_PARK_SURE", "OP_PARK_SON", "OP_APRON", "OP_TW",
           "OP_PIST", "OP_PIST_SURE", "OP_TW_TOPLAM", "OP_TW_ENUZUN", "OP_TW_SAY", "OP_N", "OP_PARK_UYUM", "OP_PIST_UYUM"]
    s = s[kol]
    for c in kol[1:]:
        s[c] = s[c].astype(float)
    if not secim:
        s.to_parquet(CIKTI, index=False)
    x = s.merge(d[["MVT_ID_mvt", "ADEP_mvt", "ay"]], on="MVT_ID_mvt")
    x["yer"] = x["OP_YER"] >= 150
    print(x.pivot_table(index="ADEP_mvt", columns="ay", values="yer", aggfunc="sum").fillna(0).astype(int).to_string())
    print(f"→ {CIKTI} ({len(s):,} satır)")


if __name__ == "__main__":
    main()

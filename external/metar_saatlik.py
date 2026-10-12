"""Ham IEM METAR dosyalarından istasyon × saat tablosu üretir.

Girdi : ../metar_ham/metar_<ICAO>_<YYYYMM>.csv  (metar_indir.py)
Çıktı : ../metar_saatlik.parquet, ../eksik_saat.csv, ../metar_ozet.txt

Bayraklar yalnızca raporun GÖZLEM kısmına uygulanır: ilk ' TEMPO', ' BECMG', ' NOSIG', ' RMK' sonrası kesilir
(bunlar tahmin ya da ek bilgidir). Saat = valid'in UTC tam saate yuvarlanması (aşağı).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "metar_ham"

KES = re.compile(r"\s(?:TEMPO|BECMG|NOSIG|RMK)\b")
BAYRAK = {  # (+ WX_RA_YOGUN)
    "WX_TS": re.compile(r"\s(?:\+|-|VC)?TS|(?:\d{3}|///)CB"),  # AUTO raporlarda yükseklik "///" olabilir
    "WX_FZ": re.compile(r"\sFZ(?:RA|DZ|FG)"),
    "WX_SN": re.compile(r"\s(?:\+|-|VC)?(?:SH|BL|DR)?(?:SN|SG|PL)\b"),
    "WX_FG": re.compile(r"\s(?:BC|MI|PR)?FG\b"),
    "WX_RA_YOGUN": re.compile(r"\s\+(?:SH)?RA"),
}
EK_BAYRAK = {"WX_TS_YALNIZ": re.compile(r"\s(?:\+|-|VC)?TS")}  # ek sütun: yalnız oraj (CB bulutu hariç)
RUZGAR = re.compile(r"\s(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT\b")
GORUS = re.compile(r"KT\s+(?:\d{3}V\d{3}\s+)?(CAVOK|\d{4})(?:NDV)?\b")
SICAKLIK = re.compile(r"\s(M?\d{2})/(M?\d{2})(?=\s|$)")
KARMA_KAR = re.compile(r"\s(?:\+|-)?(?:SH)?RASN|\s(?:\+|-)?(?:SH)?SNRA")  # WX_SN'nin kaçırdığı karışık yağış (sayım için)


def derece(s):
    if s is None or s != s:
        return np.nan
    return -float(s[1:]) if s.startswith("M") else float(s)


def ayristir(metin):
    tam = " " + str(metin).strip() + " "
    m = KES.search(tam)
    goz = tam[:m.start()] + " " if m else tam
    r = {k: bool(p.search(goz)) for k, p in {**BAYRAK, **EK_BAYRAK}.items()}
    r["_TS_TAM"] = bool(BAYRAK["WX_TS"].search(tam))          # trend dahil (karşılaştırma için)
    r["_KARMA_KAR"] = bool(KARMA_KAR.search(goz))
    w = RUZGAR.search(goz)
    r["RUZGAR_YON"] = np.nan if (w is None or w.group(1) == "VRB") else float(w.group(1))
    r["RUZGAR_KT"] = float(w.group(2)) if w else np.nan
    r["HAMLE_KT"] = float(w.group(3)) if (w and w.group(3)) else np.nan
    v = GORUS.search(goz)
    r["GORUS_M"] = np.nan if v is None else (10000.0 if v.group(1) in ("CAVOK", "9999") else float(v.group(1)))
    t = SICAKLIK.search(goz)
    r["SICAKLIK_C"] = derece(t.group(1)) if t else np.nan
    r["CIY_NOKTASI_C"] = derece(t.group(2)) if t else np.nan
    return r


def main():
    parcalar = []
    for f in sorted(HAM.glob("metar_*_*.csv")):
        icao, ay = f.stem.split("_")[1], f.stem.split("_")[2]
        d = pd.read_csv(f, usecols=["station", "valid", "metar"], dtype=str)
        d = d[d["metar"].notna() & d["valid"].notna()]
        p = pd.DataFrame([ayristir(x) for x in d["metar"]], index=d.index)
        p["ICAO"] = icao
        p["saat"] = pd.to_datetime(d["valid"]).dt.floor("h")
        p["_ay"] = ay
        parcalar.append(p)
    r = pd.concat(parcalar, ignore_index=True)
    bay = list(BAYRAK) + list(EK_BAYRAK)
    g = r.groupby(["ICAO", "saat"])
    s = g[bay + ["_TS_TAM", "_KARMA_KAR"]].max()
    s["SICAKLIK_C"] = g["SICAKLIK_C"].mean()
    s["CIY_NOKTASI_C"] = g["CIY_NOKTASI_C"].mean()
    s["GORUS_M"] = g["GORUS_M"].min()
    s["RUZGAR_KT"] = g["RUZGAR_KT"].mean()
    s["HAMLE_KT"] = g["HAMLE_KT"].max()
    s["RUZGAR_YON"] = g["RUZGAR_YON"].last()
    s["RAPOR_N"] = g.size()
    s = s.reset_index()

    # tam saat ızgarası: indirilen her istasyon-ay için
    izg = []
    for (icao, ay) in r[["ICAO", "_ay"]].drop_duplicates().itertuples(index=False):
        bas = pd.Timestamp(f"{ay[:4]}-{ay[4:]}-01")
        izg.append(pd.DataFrame({"ICAO": icao, "saat": pd.date_range(bas, bas + pd.offsets.MonthBegin(1), freq="h", inclusive="left")}))
    izg = pd.concat(izg, ignore_index=True)
    t = izg.merge(s, on=["ICAO", "saat"], how="left")
    t["RAPOR_N"] = t["RAPOR_N"].fillna(0).astype("int16")
    for c in bay:
        t[c] = t[c].astype("boolean")          # rapor yoksa <NA>
    cikti = t[["ICAO", "saat"] + bay + ["SICAKLIK_C", "CIY_NOKTASI_C", "GORUS_M", "RUZGAR_KT", "HAMLE_KT", "RUZGAR_YON", "RAPOR_N"]]
    cikti = cikti.sort_values(["ICAO", "saat"]).reset_index(drop=True)
    for c in ["SICAKLIK_C", "CIY_NOKTASI_C", "GORUS_M", "RUZGAR_KT", "HAMLE_KT", "RUZGAR_YON"]:
        cikti[c] = cikti[c].astype("float32")
    cikti.to_parquet(KOK / "metar_saatlik.parquet", index=False)

    # eksik saat tablosu (istasyon × ay, %)
    t["ay"] = t["saat"].dt.strftime("%Y-%m")
    eks = (t.assign(eksik=t["RAPOR_N"] == 0).groupby(["ICAO", "ay"])["eksik"].mean().mul(100).round(2).unstack("ay"))
    eks.to_csv(KOK / "eksik_saat.csv")

    # özet
    with open(KOK / "metar_ozet.txt", "w", encoding="utf-8") as f:
        f.write(f"rapor sayisi: {len(r)}  saat satiri: {len(cikti)}\n\n")
        f.write("Eksik saat (%), istasyon x ay:\n" + eks.to_string() + "\n\n")
        say = t.assign(ay=t["saat"].dt.strftime("%Y-%m")).groupby(["ICAO"])[bay].sum().astype(int)
        f.write("Bayrakli saat sayisi (tum aylar):\n" + say.to_string() + "\n\n")
        fark = int((s["_TS_TAM"] & ~s["WX_TS"].astype(bool)).sum())
        f.write(f"Trend (TEMPO/BECMG) kisminda TS/CB olup gozlemde olmayan saat: {fark}\n")
        f.write(f"WX_SN'nin kacirdigi karisik yagis (RASN/SNRA) iceren saat: {int((s['_KARMA_KAR'] & ~s['WX_SN'].astype(bool)).sum())}\n")
        nan = cikti[cikti.RAPOR_N > 0][["SICAKLIK_C", "CIY_NOKTASI_C", "GORUS_M", "RUZGAR_KT"]].isna().mean().mul(100).round(2)
        f.write("Raporlu saatlerde ayristirilamayan (%):\n" + nan.to_string() + "\n")
    print(open(KOK / "metar_ozet.txt", encoding="utf-8").read())


if __name__ == "__main__":
    main()

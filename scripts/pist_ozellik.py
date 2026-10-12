"""Pist konfigürasyonu özellikleri (04.10).
Ham hareketlerden (2025 eğitim + 2026 test; kalkış ve varış); hedef yalnız PC_CFG_TE'de (2025 taksisi, ay dışı):
  PC_N_DEP / PC_N_ARR : kalkışın ±30 dk'sında kalkış / iniş yapılan pist sayısı
  PC_DEGISTI          : bu konfigürasyon (kalkış + iniş pist kümesi) 1 sa önceki ile farklı mı
  PC_PAY / PC_BASKIN  : penceredeki kalkışların uçağın pistinden çıkan payı / pist en çok kullanılan mı
  PC_KUYRUK_W         : aynı pistten (AOBT_3, MVT) arasında kalkanlar, iz türbülansıyla ağırlıklı (J 2,0 H 1,6 M 1,0 L 0,7)
  PC_AGIR_N           : bunların H / J olanları
  PC_SPAN20           : havalimanında 20 önceki kalkıştan bu yana geçen süre (sn)
  PC_CFG_TE           : (havalimanı, konfigürasyon) hücresinin taksi ortalaması, ay dışı (2025) / 2025'in tamamı (2026);
                        k = 50 ile havalimanı ortalamasına doğru; taksi > 2 sa olanlar dışarıda
Çıktı: data/pist_oz.parquet"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

HAM = config.ROOT.parent / "data"   # yarışma dosyaları (2025 eğitim + test)
W = {"J": 2.0, "H": 1.6, "M": 1.0, "L": 0.7}
SUT = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "AOBT_3_flt", "WK_TBL_CAT_flt"]
parca = []
EGITIM = sorted(glob.glob(str(HAM / "training_2025-*.parquet")))
assert len(EGITIM) == 12, f"{HAM}: 12 eğitim dosyası beklenir, {len(EGITIM)} bulundu"
for f in EGITIM + [str(config.RANK_RAW)]:
    x = pd.read_parquet(f, columns=SUT + (["TAXITIME_SEC_mvt"] if f in EGITIM else []))
    x["test"] = f == str(config.RANK_RAW)
    parca.append(x)
m = pd.concat(parca, ignore_index=True)
m["apt"] = np.where(m.PHASE_mvt == "DEP", m.ADEP_mvt, m.ADES_mvt)
E0 = pd.Timestamp("1970-01-01", tz="UTC")  # pandas 3: çözünürlük µs olabilir → saniyeyi açıkça hesapla
m["t"] = (pd.to_datetime(m.MVT_TIME_UTC_mvt, utc=True) - E0).dt.total_seconds()
m["a3"] = (pd.to_datetime(m.AOBT_3_flt, utc=True) - E0).dt.total_seconds()
m["w"] = m.WK_TBL_CAT_flt.map(W).fillna(1.0)
m["agir"] = m.WK_TBL_CAT_flt.isin(["H", "J"]).astype(float)
m = m[m.RUNWAY_mvt.notna()].reset_index(drop=True)
print(f"hareket {len(m):,} (kalkış {int((m.PHASE_mvt == 'DEP').sum()):,})", flush=True)


def say(ts, a, b):
    """Sıralı ts'de [a, b] aralığına düşenlerin sayısı."""
    return np.searchsorted(ts, b, "right") - np.searchsorted(ts, a, "left")


out = []
for apt, g in m.groupby("apt"):
    dep = g[g.PHASE_mvt == "DEP"].sort_values("t")
    T = dep.t.to_numpy(float)
    pistler_d = {r: np.sort(dep.t[dep.RUNWAY_mvt == r].to_numpy(float)) for r in dep.RUNWAY_mvt.unique()}
    arr = g[g.PHASE_mvt == "ARR"]
    pistler_a = {r: np.sort(arr.t[arr.RUNWAY_mvt == r].to_numpy(float)) for r in arr.RUNWAY_mvt.unique()}
    rd, ra = sorted(pistler_d), sorted(pistler_a)

    def kume(t0):
        cd = np.stack([say(pistler_d[r], t0 - 1800, t0 + 1800) for r in rd], 1)
        ca = np.stack([say(pistler_a[r], t0 - 1800, t0 + 1800) for r in ra], 1) if ra else np.zeros((len(t0), 0))
        bit_d = (cd >= 2) @ (2.0 ** np.arange(len(rd)))
        bit_a = (ca >= 2) @ (2.0 ** np.arange(len(ra))) if ra else np.zeros(len(t0))
        return cd, ca, bit_d, bit_a

    cd, ca, bd, ba = kume(T)
    _, _, bd1, ba1 = kume(T - 3600)
    own = dep.RUNWAY_mvt.map({r: i for i, r in enumerate(rd)}).to_numpy()
    own_c = cd[np.arange(len(T)), own]
    top = cd.sum(1)
    o = pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy(), "test": dep.test.to_numpy(), "apt": apt,
                      "ay": pd.to_datetime(dep.t, unit="s").dt.month.to_numpy(),
                      "y": dep.TAXITIME_SEC_mvt.to_numpy() if "TAXITIME_SEC_mvt" in dep else np.nan,
                      "cfg": apt + "|" + pd.Series(bd).astype(int).astype(str).to_numpy() + "|" + pd.Series(ba).astype(int).astype(str).to_numpy(),
                      "PC_N_DEP": (cd >= 2).sum(1), "PC_N_ARR": (ca >= 2).sum(1) if ra else 0,
                      "PC_DEGISTI": ((bd != bd1) | (ba != ba1)).astype(float),
                      "PC_PAY": own_c / np.maximum(top, 1), "PC_BASKIN": (own_c >= cd.max(1)).astype(float)})
    # aynı pistte (AOBT_3, MVT) arasında kalkanlar, ağırlıklı
    kw, kh = np.full(len(T), np.nan), np.full(len(T), np.nan)
    a3 = dep.a3.to_numpy(float)
    for r in rd:
        i = np.where(dep.RUNWAY_mvt.to_numpy() == r)[0]
        tt, ww, hh = T[i], dep.w.to_numpy()[i], dep.agir.to_numpy()[i]
        cw, ch = np.r_[0, np.cumsum(ww)], np.r_[0, np.cumsum(hh)]
        lo = np.searchsorted(tt, np.nan_to_num(a3[i], nan=np.inf), "right")
        hi = np.searchsorted(tt, tt, "left")
        ok = np.isfinite(a3[i]) & (hi >= lo)
        kw[i] = np.where(ok, cw[np.maximum(hi, lo)] - cw[lo], np.nan)
        kh[i] = np.where(ok, ch[np.maximum(hi, lo)] - ch[lo], np.nan)
    o["PC_KUYRUK_W"], o["PC_AGIR_N"] = kw, kh
    o["PC_SPAN20"] = np.where(np.arange(len(T)) >= 20, T - np.r_[np.full(20, np.nan), T[:-20]], np.nan)
    o.loc[o.PC_SPAN20 > 21600, "PC_SPAN20"] = np.nan  # veri kümesi sınırları (ör. Temmuz 2026'nın ilk kalkışları)
    out.append(o)
    print(apt, len(o), "konfigürasyon sayısı", o.cfg.nunique(), flush=True)
o = pd.concat(out, ignore_index=True)
# PC_CFG_TE: ay dışı hedef kodlama (2025), 2026'ya 2025'in tamamı
K = 50.0
egit = o[~o.test & (o.y < 7200) & (o.y > 0)]
o["PC_CFG_TE"] = np.nan
apt_ort = egit.groupby("apt").y.mean()
for ay in range(1, 14):
    hedef = (o.test) if ay == 13 else (~o.test & (o.ay == ay))
    kay = egit if ay == 13 else egit[egit.ay != ay]
    s = kay.groupby("cfg").y.agg(["sum", "count"])
    h = o.loc[hedef, ["cfg", "apt"]].merge(s, left_on="cfg", right_index=True, how="left")
    onc = h.apt.map(kay.groupby("apt").y.mean()).to_numpy(float)
    o.loc[hedef, "PC_CFG_TE"] = (h["sum"].fillna(0).to_numpy() + K * onc) / (h["count"].fillna(0).to_numpy() + K) - onc
o["MVT_ID_mvt"] = o.MVT_ID_mvt.astype("float64")
sut = ["PC_N_DEP", "PC_N_ARR", "PC_DEGISTI", "PC_PAY", "PC_BASKIN", "PC_KUYRUK_W", "PC_AGIR_N", "PC_SPAN20", "PC_CFG_TE"]
assert o.MVT_ID_mvt.is_unique
o[["MVT_ID_mvt"] + sut].astype({c: "float32" for c in sut}).to_parquet(config.DATA_DIR / "pist_oz.parquet", index=False)
print(len(o), "\n", o[sut].describe().T[["count", "mean", "std", "min", "max"]].round(2).to_string())
print("eğitim / test ortalaması:\n", o.groupby("test")[sut].mean().round(3).T.to_string())

"""İleriye bakan / ortalanmış pencere özellikleri (01.10 gece). Gecikme endeksleri (gecikme_ozellik.py, gecikme_pist_ozellik.py)
yalnız geçmişe bakıyor; testte bütün kalkış ve varışlar bilindiği için (sonradan tahmin) uçağın kalkışından SONRAKİ trafik
de kullanılabilir: arkasında biriken kuyruk, beklediği sürenin göstergesi. Etiket (BLOCK) kullanılmaz.
Tarama (lgbh3_kqg OOF kalıntısı, 6 ay ay dışı; kontrol = GEC + pist + varış + METAR; gürültü tabanı ayda ±0,1):
  GECC −0,48 (6/6), SONRA −0,16 (6/6), NMT −0,22 (5/6); üçü birlikte −0,88 (6/6 ay, 10/10 havalimanı).
Özellikler (kalkış anı T; aynı havalimanı; kendisi hariç):
  GECC_EOBT_15 : [T − 15 dk, T + 15 dk] kalkışlarının ortanca (kalkış − EOBT_1), [−3.600, 14.400]'e kırpık
  GECF_EOBT_15 : (T, T + 15 dk] aynı ortanca (yalnız sonrakiler)
  NMT_C15 / NMT_F15 : aynı pencerelerde ortanca NM taksisi (kalkış − AOBT_3), [0, 7.200]'e kırpık
  DEP_APT_SONRA_10 / DEP_PIST_SONRA_10 : (T, T + 10 dk] havalimanından / aynı pistten kalkış sayısı
  ARR_SONRA_10 : (T, T + 10 dk] havalimanına varış sayısı
  GECC_PIST_15 / NMT_PIST_C15 : aynı pistte [T ± 15 dk] ortanca (kalkış − EOBT_1) / NM taksisi (tarama, tam paketin
                 üstüne: −0,16, 5/6 ay; LTFM −0,59)
Çıktı: data/pencere_ileri_oz.parquet
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K  # noqa: E402
from src import config  # noqa: E402

KOL = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "EOBT_1_flt", "AOBT_3_flt"]


def med_pencere(T, v, a_ofs, b_ofs):
    """Sıralı T için her i: T_j ∈ [T_i + a_ofs, T_i + b_ofs] (a_ofs > 0 ise alt uç açık), j ≠ i, v_j ortancası."""
    lo = np.searchsorted(T, T + a_ofs, "left" if a_ofs <= 0 else "right")
    hi = np.searchsorted(T, T + b_ofs, "right")
    out = np.full(len(T), np.nan)
    for i in range(len(T)):
        l, h = lo[i], hi[i]
        if h <= l:
            continue
        a = v[l:h]
        if l <= i < h:
            a = np.delete(a, i - l)
        a = a[np.isfinite(a)]
        if len(a):
            out[i] = np.median(a)
    return out


def say(s, a, b):
    """sıralı s'de a < x ≤ b olan eleman sayısı."""
    return np.searchsorted(s, b, "right") - np.searchsorted(s, a, "right")


def blok(df):
    df = df.copy()
    df["T"] = K.ep(df["MVT_TIME_UTC_mvt"])
    df["pist"] = df["RUNWAY_mvt"].astype(str)
    df["ge"] = np.clip(df["T"] - K.ep(df["EOBT_1_flt"]), -3600, 14400)
    df["nt"] = np.clip(df["T"] - K.ep(df["AOBT_3_flt"]), 0, 7200)
    dep, arr = df[df["PHASE_mvt"].eq("DEP")], df[df["PHASE_mvt"].eq("ARR")]
    parca = []
    for a, d in dep.groupby("ADEP_mvt"):
        d = d.sort_values("T", kind="stable").reset_index(drop=True)
        T, ge, nt, R = d["T"].to_numpy(float), d["ge"].to_numpy(float), d["nt"].to_numpy(float), d["pist"].to_numpy()
        o = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()})
        o["GECC_EOBT_15"] = med_pencere(T, ge, -900, 900)
        o["GECF_EOBT_15"] = med_pencere(T, ge, 0, 900)
        o["NMT_C15"] = med_pencere(T, nt, -900, 900)
        o["NMT_F15"] = med_pencere(T, nt, 0, 900)
        o["DEP_APT_SONRA_10"] = say(T, T, T + 600).astype(float)
        o["DEP_PIST_SONRA_10"] = np.nan
        o["GECC_PIST_15"] = np.nan
        o["NMT_PIST_C15"] = np.nan
        for r in np.unique(R):
            i = np.where(R == r)[0]
            o.loc[i, "DEP_PIST_SONRA_10"] = say(T[i], T[i], T[i] + 600)
            o.loc[i, "GECC_PIST_15"] = med_pencere(T[i], ge[i], -900, 900)
            o.loc[i, "NMT_PIST_C15"] = med_pencere(T[i], nt[i], -900, 900)
        at = np.sort(arr.loc[arr["ADES_mvt"].eq(a), "T"].to_numpy(float))
        o["ARR_SONRA_10"] = say(at, T, T + 600).astype(float)
        parca.append(o)
    return pd.concat(parca, ignore_index=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                             "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=KOL)
    out = pd.concat([blok(tr), blok(rk)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "pencere_ileri_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T[["count", "mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()

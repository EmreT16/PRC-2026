"""Model B için park yeri dizisi ve komşu taksi özellikleri (07.10 gece). Soru: B uçuşları sıralı mı, önceki /
sonraki uçuşu görerek tahmin edilebilir mi?

NM'siz (B) satırlarda AOBT_3 yok → park_varis_oz'un IN_TURN / IN_TAMPON'u B'de %0 dolu, referansı da (kalkış − 15 dk)
kaba. Burada her şey kalkış anına (T = MVT_TIME) göre ve yalnız etiketsiz bilgiyle (varışların park anı eğitimde ve testte
açık; kalkışın kendi BLOCK'u kullanılmaz):
  PB_ON1      : T − aynı park yerine T'den önce yanaşan son varışın park anı (sn; 12 sa pencere)
  PB_ON1_HY   : o varışın işletmecisi kalkışınkiyle aynı mı (1/0) — aynı uçak olma olasılığı (işletmeci FLIGHT_mvt'den;
                NM'siz kalkışta IATA → ICAO, data/iata_icao.json)
  PB_ON1_TIP  : uçak tipi aynı mı (1/0)
  PB_ON1_GEC  : o varışın gecikmesi (iniş − tarife), [−3.600, 14.400]
  PB_ON2      : T − sondan ikinci varışın park anı
  PB_AYNI     : T − aynı park yerine yanaşan, işletmecisi ve tipi aynı son varışın park anı (büyük olasılıkla kendi gelişi)
  PB_SON1     : aynı park yerine T'den sonra yanaşan ilk varışın park anı − T
  PD_ON1      : T − aynı park yerinden önceki kalkışın T'si;  PD_SON1: sonraki kalkışın T'si − T
  PB_KT15     : ±15 dk içinde aynı havalimanından kalkan NM'li (Model A) satırların tahmin edilen taksisinin ortalaması
                (kendisi hariç; eğitim: kqgpPe OOF, 12 ay; test: TEST_D_kqgpPe)  PB_KT15_N: o satır sayısı
  PB_KT15_P   : aynı, aynı pistten
Fiziksel anlam: kalkan uçak park yerini bir sonraki uçak yanaşmadan önce boşaltmış olmalı (off-block ≤ sonraki varış) ve
kendi gelişinden sonra çıkmış olmalı (off-block ≥ kendi varışı) → taksi süresi için etiketsiz alt / üst sınır ipuçları.
Çıktı: data/park_b_oz.parquet (eğitim + test tüm kalkışlar)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PRC_HARMAN", "kqgpPe")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K  # noqa: E402
from src import adsb as A, config  # noqa: E402

KOL = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
       "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt"]
PENCERE = 12 * 3600.0


IATA_ICAO = None


def isletmeci(df):
    """İşletmeci (ICAO 3 harf), her satırda olan FLIGHT_mvt'den: NM'li kalkış ve varışlarda ICAO çağrı kodu (DLH2P),
    NM'siz kalkışlarda IATA uçuş numarası (LH8318) → data/iata_icao.json. AIRCRAFT_OPERATOR_flt NM'siz satırlarda yok
    (A ile B arasında tutarsız olur), kullanılmaz."""
    global IATA_ICAO
    if IATA_ICAO is None:
        import json
        IATA_ICAO = json.loads((config.DATA_DIR / "iata_icao.json").read_text(encoding="utf-8"))
    f = df["FLIGHT_mvt"].fillna("").astype(str).str.upper()
    icao = f.str.extract(r"^([A-Z]{3})(?=[0-9])")[0]
    iata = f.str.extract(r"^([A-Z0-9]{2})(?=[0-9])")[0].map(IATA_ICAO)
    return icao.fillna(iata).fillna("?").astype(str)


def asof(sol, sag, by, yon, izin=True):
    return pd.merge_asof(sol.sort_values("T"), sag.sort_values("B"), left_on="T", right_on="B", by=by, direction=yon,
                         tolerance=PENCERE, allow_exact_matches=izin)


def blok(df, tah):
    dep = df[df["PHASE_mvt"].eq("DEP")].copy()
    arr = df[df["PHASE_mvt"].eq("ARR")].copy()
    dep["apt"], arr["apt"] = dep["ADEP_mvt"], arr["ADES_mvt"]
    dep["T"] = K.ep(dep["MVT_TIME_UTC_mvt"])
    arr["B"] = K.ep(arr["BLOCK_TIME_UTC_mvt"])
    arr["g"] = np.clip(K.ep(arr["MVT_TIME_UTC_mvt"]) - K.ep(arr["SCHED_TIME_UTC_mvt"]), -3600, 14400)
    dep["op"], arr["op"] = isletmeci(dep), isletmeci(arr)
    dep["tip"], arr["tip"] = dep["AIRCRAFT_TYPE_mvt"].fillna("?").astype(str), arr["AIRCRAFT_TYPE_mvt"].fillna("?").astype(str)
    arr = arr[np.isfinite(arr["B"]) & arr["STAND_mvt"].notna()]
    out = dep[["MVT_ID_mvt"]].copy()
    d = dep[dep["STAND_mvt"].notna() & np.isfinite(dep["T"])][["MVT_ID_mvt", "apt", "STAND_mvt", "T", "op", "tip"]]
    a = arr[["apt", "STAND_mvt", "B", "g", "op", "tip"]].rename(columns={"op": "a_op", "tip": "a_tip"})
    o1 = asof(d, a, ["apt", "STAND_mvt"], "backward", izin=False)
    r = pd.DataFrame({"MVT_ID_mvt": o1["MVT_ID_mvt"], "PB_ON1": o1["T"] - o1["B"],
                      "PB_ON1_HY": np.where(o1["B"].notna(), (o1["a_op"] == o1["op"]).astype(float), np.nan),
                      "PB_ON1_TIP": np.where(o1["B"].notna(), (o1["a_tip"] == o1["tip"]).astype(float), np.nan),
                      "PB_ON1_GEC": o1["g"]})
    # sondan ikinci varış: son varışın park anından hemen önceye göre
    d2 = o1[["MVT_ID_mvt", "apt", "STAND_mvt", "T", "B"]].dropna(subset=["B"]).rename(columns={"T": "T0", "B": "T"})
    o2 = asof(d2, a[["apt", "STAND_mvt", "B"]], ["apt", "STAND_mvt"], "backward", izin=False)
    r = r.merge(pd.DataFrame({"MVT_ID_mvt": o2["MVT_ID_mvt"], "PB_ON2": o2["T0"] - o2["B"]}), on="MVT_ID_mvt", how="left")
    a3 = arr[["apt", "STAND_mvt", "op", "tip", "B"]]
    o3 = asof(d, a3, ["apt", "STAND_mvt", "op", "tip"], "backward", izin=False)
    r = r.merge(pd.DataFrame({"MVT_ID_mvt": o3["MVT_ID_mvt"], "PB_AYNI": o3["T"] - o3["B"]}), on="MVT_ID_mvt", how="left")
    o4 = asof(d, a[["apt", "STAND_mvt", "B"]], ["apt", "STAND_mvt"], "forward", izin=False)
    r = r.merge(pd.DataFrame({"MVT_ID_mvt": o4["MVT_ID_mvt"], "PB_SON1": o4["B"] - o4["T"]}), on="MVT_ID_mvt", how="left")
    # aynı park yerinden önceki / sonraki kalkış
    s = d.sort_values(["apt", "STAND_mvt", "T"])
    g = s.groupby(["apt", "STAND_mvt"])["T"]
    s["PD_ON1"], s["PD_SON1"] = s["T"] - g.shift(1), g.shift(-1) - s["T"]
    s.loc[s["PD_ON1"] > PENCERE, "PD_ON1"] = np.nan
    s.loc[s["PD_SON1"] > PENCERE, "PD_SON1"] = np.nan
    r = r.merge(s[["MVT_ID_mvt", "PD_ON1", "PD_SON1"]], on="MVT_ID_mvt", how="left")
    # komşu NM'li kalkışların tahmin edilen taksisi (±15 dk)
    k = dep[["MVT_ID_mvt", "apt", "RUNWAY_mvt", "T"]].merge(tah, on="MVT_ID_mvt", how="left")
    kt = {}
    for ad, anah in (("PB_KT15", ["apt"]), ("PB_KT15_P", ["apt", "RUNWAY_mvt"])):
        v, n = np.full(len(k), np.nan), np.zeros(len(k))
        for _, idx in k.groupby(anah, dropna=False).indices.items():
            t = k["T"].to_numpy(float)[idx]
            o = np.argsort(t)
            idx, t = idx[o], t[o]
            x = k["tah"].to_numpy(float)[idx]
            ok = np.isfinite(x)
            cs, cn = np.concatenate([[0], np.cumsum(np.where(ok, x, 0))]), np.concatenate([[0], np.cumsum(ok)])
            lo, hi = np.searchsorted(t, t - 900, "left"), np.searchsorted(t, t + 900, "right")
            top, say = cs[hi] - cs[lo] - np.where(ok, x, 0), cn[hi] - cn[lo] - ok
            v[idx] = np.where(say > 0, top / np.maximum(say, 1), np.nan)
            n[idx] = say
        kt[ad], kt[ad + "_N"] = v, n
    r = r.merge(pd.DataFrame({"MVT_ID_mvt": k["MVT_ID_mvt"].to_numpy(), **kt}), on="MVT_ID_mvt", how="left")
    return out.merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one")


def tahminler():
    d = pd.concat([A.d_dw_split(f"split_{s}") for s in range(1, 7)], ignore_index=True)
    e = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "tah": np.maximum(60.0, d["NM_TAXI_DIFF_SEC"] - d["ofset_taban"])})
    t = pd.read_parquet(A.PRED / f"TEST_{A.TEST_AD}.parquet", columns=["MVT_ID_mvt", "taksi"]).rename(columns={"taksi": "tah"})
    return pd.concat([e, t], ignore_index=True).drop_duplicates("MVT_ID_mvt")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tah = tahminler()
    print(f"komşu tahmini: {len(tah):,} NM'li satır ({time.time() - t0:.0f} sn)", flush=True)
    tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                             "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=KOL)
    out = pd.concat([blok(tr, tah), blok(rk, tah)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "park_b_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T[["count", "mean", "50%", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()

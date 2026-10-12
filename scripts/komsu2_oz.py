"""Komşu özellikleri, 2. tur (03.10 akşam; v202 LB −0,99 sonrası). Komşuların tahmini artık komşulu tabandan (kqgpy).
Gruplar (ozellik(); tarama_olay.py TARAMA_K2=1 de aynı işlevi kullanır):
  K2    : KTAH'ın kqgpy'den yeniden hesabı (önceki/sonraki 10 kalkış havalimanında, 5 aynı pistte; kendisi hariç)
  K2OFS : komşuların tahmini ofseti (NM − tahmin) ortalaması (havalimanı 10, pist 5)
  K2V   : havalimanında k = 3 / 20 ortalaması, k = 10 standart sapması
  K2Q   : kendi tahmini taksi süresi içinde (T − tah, T] aynı pistten / havalimanından kalkan uçuş sayısı (kuyruk önü)
Ana kip: eğitim satırları kqgpy'nin ay dışı OOF'u (split_1…6 = 12 ay), test TEST_D_kqgpy → data/komsu2_oz.parquet.
Kullanım: python scripts/komsu2_oz.py [GRUP1,GRUP2,...]   (varsayılan: hepsi)"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRUPLAR = {"K2": ["K2_ORT", "K2_FARK", "K2_P_ORT", "K2_P_FARK"], "K2OFS": ["K2_OFS", "K2_P_OFS"],
           "K2V": ["K2_ORT3", "K2_ORT20", "K2_STD"], "K2Q": ["K2_Q_P", "K2_Q_A"]}


def _kom(v, k):
    """Sıralı dizide önceki/sonraki k elemanın (kendisi hariç) ortalaması ve standart sapması."""
    v = np.nan_to_num(v.astype(float))
    cs, cs2, n = np.r_[0.0, np.cumsum(v)], np.r_[0.0, np.cumsum(v * v)], len(v)
    i = np.arange(n)
    lo, hi = np.maximum(i - k, 0), np.minimum(i + k + 1, n)
    c = hi - lo - 1
    s, s2 = cs[hi] - cs[lo] - v, cs2[hi] - cs2[lo] - v * v
    m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    sd = np.where(c > 1, np.sqrt(np.maximum(s2 / np.maximum(c, 1) - m * m, 0)), np.nan)
    return m, sd


def ozellik(x):
    """x: ADEP_mvt, KOM_T (kalkış anı, sn), KOM_PIST, nm, tah (tahmini taksi). Dönüş: GRUPLAR sütunlarıyla DataFrame."""
    tah, nm, T = x["tah"].to_numpy(float), x["nm"].to_numpy(float), x["KOM_T"].to_numpy(float)
    ofs = nm - tah
    out = {c: np.full(len(x), np.nan) for g in GRUPLAR.values() for c in g}
    for anah, pist in ((["ADEP_mvt"], False), (["ADEP_mvt", "KOM_PIST"], True)):
        for _, idx in x.groupby(anah, observed=True).indices.items():
            o = idx[np.argsort(T[idx], kind="stable")]
            tt, ta = T[o], tah[o]
            q = np.searchsorted(tt, tt, "left") - np.searchsorted(tt, tt - ta, "right")  # (T − tah, T) içinde kalkan
            if pist:
                m, _ = _kom(ta, 5)
                out["K2_P_ORT"][o], out["K2_P_FARK"][o] = m, ta - m
                out["K2_P_OFS"][o] = _kom(ofs[o], 5)[0]
                out["K2_Q_P"][o] = q
            else:
                m, sd = _kom(ta, 10)
                out["K2_ORT"][o], out["K2_FARK"][o], out["K2_STD"][o] = m, ta - m, sd
                out["K2_OFS"][o] = _kom(ofs[o], 10)[0]
                out["K2_ORT3"][o], out["K2_ORT20"][o] = _kom(ta, 3)[0], _kom(ta, 20)[0]
                out["K2_Q_A"][o] = q
    return pd.DataFrame(out, index=x.index)


def main(gruplar):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    os.environ["PRC_HARMAN"] = "kqgpy"
    import adsb as A
    from src import config
    d = pd.concat([A.d_dw_split(f"split_{k}") for k in range(1, 7)], ignore_index=True)
    d["tah"] = np.maximum(60, d["NM_TAXI_DIFF_SEC"] - d["ofset_taban"])
    t = pd.read_parquet(A.PRED / "TEST_D_kqgpy.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "taksi"])
    t = t[t["NM_TAXI_DIFF_SEC"].notna()].rename(columns={"taksi": "tah"})
    k = pd.read_parquet(config.DATA_DIR / "komsu_oz.parquet", columns=["MVT_ID_mvt", "KOM_T", "KOM_PIST"])
    base = pd.concat([pd.read_parquet(config.DATA_DIR / f"base_v126_{s}_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"])
                      for s in ("train", "rank")])
    sut = [c for g in gruplar for c in GRUPLAR[g]]
    out = []
    for x in (d[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "tah"]], t[["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "tah"]]):
        x = x.rename(columns={"NM_TAXI_DIFF_SEC": "nm"}).merge(k, on="MVT_ID_mvt", how="left").merge(base, on="MVT_ID_mvt", how="left")
        out.append(pd.concat([x[["MVT_ID_mvt"]], ozellik(x)[sut]], axis=1))
    o = pd.concat(out, ignore_index=True)
    assert o["MVT_ID_mvt"].is_unique
    o.astype({c: "float32" for c in sut}).to_parquet(config.DATA_DIR / "komsu2_oz.parquet", index=False)
    print(len(o), o[sut].describe().T[["count", "mean", "std"]].round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else list(GRUPLAR))

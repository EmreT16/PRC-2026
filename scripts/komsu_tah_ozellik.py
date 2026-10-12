"""Komşu kalkışların tahmin edilen taksisi — tüm aylar (03.10; Model A içine özellik olarak).
Eğitim satırları: kqgpxg harmanının ay dışı OOF'u (split_1…6 = 12 ay); test: TEST_D_kqgpxg. komsu_ikinci.py'deki KTAH
tanımı (önceki/sonraki 10 kalkış, aynı pistte 5; kendisi hariç). Çıktı: data/komsu_tah_oz.parquet"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["PRC_HARMAN"] = "kqgpxg"
import adsb as A  # noqa: E402
from komsu_ikinci import komsu_ort  # noqa: E402
from src import config  # noqa: E402

d = pd.concat([A.d_dw_split(f"split_{k}") for k in range(1, 7)], ignore_index=True)
d["tah"] = np.maximum(60, d["NM_TAXI_DIFF_SEC"] - d["ofset_taban"])
t = pd.read_parquet(A.PRED / "TEST_D_kqgpxg.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "taksi"]).rename(columns={"taksi": "tah"})
t = t[t["NM_TAXI_DIFF_SEC"].notna()]
k = pd.read_parquet(config.DATA_DIR / "komsu_oz.parquet", columns=["MVT_ID_mvt", "KOM_T", "KOM_PIST"])
base = pd.concat([pd.read_parquet(config.DATA_DIR / f"base_v126_{s}_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]) for s in ("train", "rank")])
out = []
for x in (d[["MVT_ID_mvt", "tah"]], t[["MVT_ID_mvt", "tah"]]):
    x = x.merge(k, on="MVT_ID_mvt", how="left").merge(base, on="MVT_ID_mvt", how="left")
    tah, Tk = x["tah"].to_numpy(float), x["KOM_T"].to_numpy(float)
    for ad, anah, kk in (("KTAH", ["ADEP_mvt"], 10), ("KTAH_P", ["ADEP_mvt", "KOM_PIST"], 5)):
        ort = np.full(len(x), np.nan)
        for _, idx in x.groupby(anah).indices.items():
            o = idx[np.argsort(Tk[idx], kind="stable")]
            ort[o] = komsu_ort(tah[o], kk)
        x[f"{ad}_ORT"], x[f"{ad}_FARK"] = ort, tah - ort
    out.append(x[["MVT_ID_mvt", "KTAH_ORT", "KTAH_FARK", "KTAH_P_ORT", "KTAH_P_FARK"]])
o = pd.concat(out, ignore_index=True)
assert o["MVT_ID_mvt"].is_unique
o.astype({c: "float32" for c in o.columns[1:]}).to_parquet(config.DATA_DIR / "komsu_tah_oz.parquet", index=False)
print(len(o), o.describe().T[["count", "mean", "std"]].round(1).to_string())

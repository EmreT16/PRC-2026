"""Hava olayı / yapay NM / gün oranı özellikleri (30.09 gece).

Kaynaklar (salt okuma): extra_data/metar_saatlik.parquet (IEM ASOS METAR, kamu malı; yalnız gözlem kısmı),
kutuk/yapay_nm_kumeleri.csv (karar == "yapay" olan NM taksi kümeleri; scripts/yapay_nm.py, yalnız 2025 eğitim verisi).
Özellikler (kalkış saati h = MVT'nin UTC saati):
  WX_TSY_0/1, WX_TS_0/1, WX_TS_S1 (h+1), WX_FZ_0/1, WX_SN_0/1, WX_SN_3S (h−3…h), WX_FG_0, WX_RAY_0,
  WX_SPREAD (sıcaklık − çiy noktası), WX_GORUS, WX_RUZGAR, WX_HAMLE (h anı)
  NM_YAPAY (NM taksi yapay kümede mi), NM_YAPAY_MERKEZ (kümenin merkezi, sn)
  GUN_ORAN = günün kalkış sayısı ÷ aynı (havalimanı, ay, haftanın günü) ortancası (iptal / kapasite vekili)
Çıktı: data/wx_oz.parquet (MVT_ID_mvt + özellikler; 2025'in 12 ayı ve 2026 testi).
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

VERI = config.EK_VERI
KOL = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]


def metar():
    m = pd.read_parquet(VERI / "metar_saatlik.parquet")
    m["saat"] = pd.to_datetime(m["saat"]).dt.tz_localize("UTC") if m["saat"].dt.tz is None else m["saat"]
    for c in [c for c in m.columns if c.startswith("WX_")]:
        m[c] = m[c].astype("float32")
    m["SPREAD"] = (m["SICAKLIK_C"] - m["CIY_NOKTASI_C"]).astype("float32")
    return m.set_index(["ICAO", "saat"])


def blok(df, m, kume):
    df = df[df["PHASE_mvt"].eq("DEP")].copy()
    h = df["MVT_TIME_UTC_mvt"].dt.floor("h")
    out = pd.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"].to_numpy()})

    def al(kol, kayma):
        idx = pd.MultiIndex.from_arrays([df["ADEP_mvt"].astype(str).to_numpy(), (h + pd.Timedelta(hours=kayma)).to_numpy()])
        return m[kol].reindex(idx).to_numpy(float)

    for ad, kol, kaymalar in (("TSY", "WX_TS_YALNIZ", (0, -1)), ("TS", "WX_TS", (0, -1)), ("FZ", "WX_FZ", (0, -1)),
                              ("SN", "WX_SN", (0, -1)), ("FG", "WX_FG", (0,)), ("RAY", "WX_RA_YOGUN", (0,))):
        for k in kaymalar:
            out[f"WX_{ad}_{-k}"] = al(kol, k)
    out["WX_TS_S1"] = al("WX_TS", 1)
    out["WX_SN_3S"] = np.nanmax(np.column_stack([al("WX_SN", -k) for k in range(4)]), axis=1)
    for ad, kol in (("SPREAD", "SPREAD"), ("GORUS", "GORUS_M"), ("RUZGAR", "RUZGAR_KT"), ("HAMLE", "HAMLE_KT")):
        out[f"WX_{ad}"] = al(kol, 0)
    nm = (K.ep(df["MVT_TIME_UTC_mvt"]) - K.ep(df["AOBT_3_flt"]))
    yapay, merkez = np.zeros(len(df)), np.full(len(df), np.nan)
    apt = df["ADEP_mvt"].to_numpy()
    for r in kume.itertuples(index=False):
        i = (apt == r.ICAO) & (nm >= r.alt) & (nm <= r.ust)
        yapay[i], merkez[i] = 1.0, r.merkez
    out["NM_YAPAY"] = np.where(np.isfinite(nm), yapay, np.nan)
    out["NM_YAPAY_MERKEZ"] = merkez
    gun = df["MVT_TIME_UTC_mvt"].dt.floor("D")
    g = df.assign(gun=gun).groupby(["ADEP_mvt", "gun"]).size().rename("n").reset_index()
    g["ay"], g["hg"] = g["gun"].dt.month, g["gun"].dt.dayofweek
    g["med"] = g.groupby(["ADEP_mvt", "ay", "hg"])["n"].transform("median")
    g["GUN_ORAN"] = g["n"] / g["med"]
    out["GUN_ORAN"] = df.assign(gun=gun).merge(g[["ADEP_mvt", "gun", "GUN_ORAN"]], on=["ADEP_mvt", "gun"], how="left")["GUN_ORAN"].to_numpy()
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    m = metar()
    kume = pd.read_csv(config.LOG_DIR / "yapay_nm_kumeleri.csv")
    kume = kume[kume["karar"].astype(str).str.strip().eq("yapay")][["ICAO", "merkez", "alt", "ust"]]
    print("yapay kümeler:", kume.groupby("ICAO").size().to_dict())
    tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{a:02d}-01_2025-{a + 1:02d}-01.parquet" if a < 12 else
                                             "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for a in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=KOL)
    out = pd.concat([blok(tr, m, kume), blok(rk, m, kume)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    for c in out.columns[1:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(config.DATA_DIR / "wx_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T[["count", "mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()

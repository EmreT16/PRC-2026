"""Aynı park yerindeki önceki / sonraki varış özellikleri (01.10 gece).
Önceki varış (off-block referansından önce aynı park yerine yanaşan son uçak; çoğunlukla kalkan uçağın kendisi)
ve sonraki varış (referanstan sonra yanaşan ilk uçak: park yeri o andan önce boşalmış
olmalı, yani gerçek off-block için üst sınır). Varışların BLOCK (park) anı eğitimde ve testte açık (ranking.parquet'te %100).
Kalkışın kendi BLOCK'u (etiket) kullanılmaz. Referans = AOBT_3 (NM off-block), yoksa kalkış − 15 dk.
Tarama (lgbh3_kqg OOF kalıntısı, 6 ay ay dışı; kontrol = tam _kqgp paketi; gürültü tabanı ±0,1):
  önceki varış −0,59 (6/6; EGLL −1,07, LFPG −1,05, LTFM −1,20); sonraki varış −0,17 (6/6); ikisi birlikte −0,80 (6/6).
Özellikler (sn):
  IN_GEC    : önceki varışın gecikmesi (iniş − tarife), [−3.600, 14.400]
  IN_TURN   : AOBT_3 − önceki varışın park anı (dönüş süresi)
  IN_TAMPON : EOBT_1 − önceki varışın park anı (planlanan dönüş)
  NX_GAP    : sonraki varışın park anı − referans
  NX_T      : sonraki varışın park anı − kalkış anı
Eşleşme penceresi 6 saat. Çıktı: data/park_varis_oz.parquet
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

KOL = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
       "STAND_mvt", "EOBT_1_flt", "AOBT_3_flt"]
PENCERE = 6 * 3600.0


def blok(df):
    dep = df[df["PHASE_mvt"].eq("DEP")].copy()
    arr = df[df["PHASE_mvt"].eq("ARR")].copy()
    print(f"  varış BLOCK dolu: {arr['BLOCK_TIME_UTC_mvt'].notna().mean():.3f}", flush=True)
    arr["apt"], dep["apt"] = arr["ADES_mvt"], dep["ADEP_mvt"]
    arr["B"] = K.ep(arr["BLOCK_TIME_UTC_mvt"])
    arr["g"] = np.clip(K.ep(arr["MVT_TIME_UTC_mvt"]) - K.ep(arr["SCHED_TIME_UTC_mvt"]), -3600, 14400)
    dep["O"], dep["T"], dep["E"] = K.ep(dep["AOBT_3_flt"]), K.ep(dep["MVT_TIME_UTC_mvt"]), K.ep(dep["EOBT_1_flt"])
    dep["ref"] = np.where(np.isfinite(dep["O"]), dep["O"], dep["T"] - 900.0)
    arr = arr[np.isfinite(arr["B"]) & arr["STAND_mvt"].notna()].sort_values("B")
    tum = dep[["MVT_ID_mvt"]]
    dep = dep[dep["STAND_mvt"].notna()].sort_values("ref")
    on = pd.merge_asof(dep[["MVT_ID_mvt", "apt", "STAND_mvt", "ref", "O", "E"]], arr[["apt", "STAND_mvt", "B", "g"]],
                       left_on="ref", right_on="B", by=["apt", "STAND_mvt"], direction="backward", tolerance=PENCERE)
    so = pd.merge_asof(dep[["MVT_ID_mvt", "apt", "STAND_mvt", "ref", "T"]], arr[["apt", "STAND_mvt", "B"]],
                       left_on="ref", right_on="B", by=["apt", "STAND_mvt"], direction="forward",
                       allow_exact_matches=False, tolerance=PENCERE)
    out = pd.DataFrame({"MVT_ID_mvt": on["MVT_ID_mvt"].to_numpy(), "IN_GEC": on["g"].to_numpy(float),
                        "IN_TURN": (on["O"] - on["B"]).to_numpy(float), "IN_TAMPON": (on["E"] - on["B"]).to_numpy(float)})
    so = so.assign(NX_GAP=so["B"] - so["ref"], NX_T=so["B"] - so["T"])[["MVT_ID_mvt", "NX_GAP", "NX_T"]]
    out = out.merge(so, on="MVT_ID_mvt", how="left", validate="one_to_one")
    return tum.merge(out, on="MVT_ID_mvt", how="left", validate="one_to_one")  # park yeri boş kalkışlar NaN


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
    out.to_parquet(config.DATA_DIR / "park_varis_oz.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T[["count", "mean", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()

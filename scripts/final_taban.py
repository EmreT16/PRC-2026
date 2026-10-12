"""Taban tablolar: PRC_2 özellik hattı (dep_train / dep_rank) + PRC_4 V3 test özellikleri (08.10).

Neden: v126 tabanının iki dış parçası salt okunur klasörlerde üretilmişti: PRC_2/src/build_dataset.py (12 ay eğitim +
test, DEP + ARR birlikte → 117 özellik; trafik pencereleri ay sınırını aşar, park–pist / operatör grup istatistikleri
eğitim + test birlikte hesaplanır) ve PRC_4/src/feature_engineering_v3.py (pist fiziksel boyutları + NM zaman farkları).
Final aşamasında test = final_ranking.parquet (Oca + Şub + Haz + Tem 2026) olduğu için ikisi de yeniden çalışmalı.
PRC_2 modülleri src/prc2_taban/'a kopyalandı (yalnız yorumlar düzenlendi) (features, airport_info, traffic_features, group_stats,
weather_merge, holiday_features); hava girdileri PRC_2/data/weather_all_v2.parquet ve dest_weather.parquet (2026'nın 1–7.
aylarını saatlik kapsıyor).
V3'ün STAND_RWY_TE_OFFSET sütunu tabanda kullanılmıyor (features_v126.base_table atıyor; TE katlarda yeniden hesaplanır) →
NaN yazılır.
Kullanım: python scripts/final_taban.py kontrol         → eski ranking.parquet ile PRC_2 dep_train / dep_rank'i yeniden üretir,
                                                          dosya yazmadan karşılaştırır (taşımanın doğrulaması)
          PRC_FINAL=1 python scripts/final_taban.py     → data_final/dep_train.parquet, dep_rank.parquet,
                                                          advanced_features_rank_v3.parquet + eski tablolarla karşılaştırma
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "prc2_taban"))
from src import config  # noqa: E402
from features import build_features  # noqa: E402  (src/prc2_taban)

HAM = config.PRC2_DIR / "data"
RWY = config.PRC4_DIR / "data" / "features" / "external_runways.csv"
APT10 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def dep_tablolari(rank_yolu):
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(HAM / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                           "training_2025-12-01_2026-01-01.parquet")) for m in range(1, 13)], ignore_index=True)
    rk = pd.read_parquet(rank_yolu)
    tr_id = set(tr.loc[tr["PHASE_mvt"].eq("DEP"), "MVT_ID_mvt"])
    rk_id = set(rk.loc[rk["PHASE_mvt"].eq("DEP"), "MVT_ID_mvt"])
    print(f"ham: eğitim {len(tr):,} (DEP {len(tr_id):,}) | test {len(rk):,} (DEP {len(rk_id):,}) — {rank_yolu.name}", flush=True)
    f = build_features(pd.concat([tr, rk], ignore_index=True), str(HAM / "weather_all_v2.parquet"), str(HAM / "dest_weather.parquet"))
    dtr = f[f["MVT_ID_mvt"].isin(tr_id)].reset_index(drop=True)
    drk = f[f["MVT_ID_mvt"].isin(rk_id)].reset_index(drop=True)
    assert len(dtr) == len(tr_id) and len(drk) == len(rk_id)
    print(f"özellikler: {f.shape[1]} sütun | {time.time() - t0:.0f} sn", flush=True)
    return dtr, drk


def v3_test(drk):
    """PRC_4/src/feature_engineering_v3.py'nin test kısmı (pist haritaları + NM zaman farkları), aynı kurallarla."""
    r = pd.read_csv(RWY)
    r = r[r["airport_ident"].isin(APT10)]
    harita = {}
    for _, x in r.iterrows():
        for uc in ("le", "he"):
            if pd.notna(x[f"{uc}_ident"]):
                ident = str(x[f"{uc}_ident"]).strip()
                deger = (x["length_ft"], x[f"{uc}_heading_degT"], x[f"{uc}_displaced_threshold_ft"])
                harita[(x["airport_ident"], ident)] = deger
                if ident.startswith("0"):
                    harita[(x["airport_ident"], ident[1:])] = deger
    v = [harita.get((a, str(w).strip()), (np.nan, np.nan, np.nan)) for a, w in zip(drk["ADEP_mvt"], drk["RUNWAY_mvt"])]
    z = lambda c: drk[c].fillna(0).astype(np.float32)  # noqa: E731
    ao, eo, lo, io, a3, sc = (z(c) for c in ("AOBT_DIFF_SEC", "EOBT_DIFF_SEC", "LOBT_DIFF_SEC", "IOBT_DIFF_SEC",
                                              "ARVT3_DIFF_SEC", "SCHED_DIFF_SEC"))
    return pd.DataFrame({"MVT_ID_mvt": drk["MVT_ID_mvt"].to_numpy(),
                         "RWY_PHYSICAL_LENGTH_FT": np.array([x[0] for x in v], dtype=np.float32),
                         "RWY_HEADING_DEG_T": np.array([x[1] for x in v], dtype=np.float32),
                         "RWY_DISPLACED_THRESH_FT": np.array([x[2] for x in v], dtype=np.float32),
                         "STAND_RWY_TE_OFFSET": np.float32(np.nan),
                         "AOBT_MINUS_EOBT_v3": ao - eo, "AOBT_MINUS_LOBT_v3": ao - lo, "AOBT_MINUS_IOBT_v3": ao - io,
                         "EOBT_MINUS_IOBT_v3": eo - io, "EOBT_MINUS_LOBT_v3": eo - lo, "ARVT3_MINUS_AOBT_v3": a3 - ao,
                         "SCHED_MINUS_AOBT_v3": sc - ao})


def karsilastir(ad, yeni, eski, esik=1e-4):
    """Ortak satırlarda sütun sütun fark: farklı satır payı ve en büyük mutlak fark (sayısal)."""
    m = eski.merge(yeni, on="MVT_ID_mvt", suffixes=("_e", "_y"))
    print(f"\n{ad}: eski {len(eski):,} | yeni {len(yeni):,} | ortak {len(m):,} | sütun eski {eski.shape[1]} yeni {yeni.shape[1]}"
          f"{' | yalnız eski: ' + str(sorted(set(eski.columns) - set(yeni.columns))) if set(eski.columns) - set(yeni.columns) else ''}"
          f"{' | yalnız yeni: ' + str(sorted(set(yeni.columns) - set(eski.columns))) if set(yeni.columns) - set(eski.columns) else ''}")
    satir = []
    for c in [c for c in eski.columns if c != "MVT_ID_mvt" and c in yeni.columns]:
        a, b = m[c + "_e"], m[c + "_y"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            fark = ~((np.abs(a - b) <= esik * np.maximum(1, np.abs(a))) | (np.isnan(a) & np.isnan(b)))
            buyuk = float(np.nanmax(np.abs(a - b))) if fark.any() else 0.0
        else:
            fark = (a.astype(str) != b.astype(str)).to_numpy()
            buyuk = np.nan
        if fark.any():
            satir.append((c, round(100 * fark.mean(), 3), buyuk))
    if satir:
        print(pd.DataFrame(satir, columns=["sütun", "farklı %", "maks |fark|"]).sort_values("farklı %", ascending=False).to_string(index=False))
    else:
        print("  tüm sütunlar aynı")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    kontrol = "kontrol" in sys.argv[1:]
    if kontrol:
        assert not config.FINAL, "kontrol normal modda çalışır"
        dtr, drk = dep_tablolari(config.PRC2_DIR / "data" / "ranking.parquet")
        karsilastir("dep_train (yeniden ↔ PRC_2)", dtr, pd.read_parquet(config.DEP_TRAIN))
        karsilastir("dep_rank (yeniden ↔ PRC_2)", drk, pd.read_parquet(config.DEP_RANK))
        karsilastir("V3 test (yeniden ↔ PRC_4)", v3_test(drk).drop(columns=["STAND_RWY_TE_OFFSET"]),
                    pd.read_parquet(config.PRC4_DIR / "data" / "features" / "advanced_features_rank_v3.parquet")
                    .drop(columns=["STAND_RWY_TE_OFFSET"]))
        return
    assert config.FINAL, "final tabloları için PRC_FINAL=1"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dtr, drk = dep_tablolari(config.RANK_RAW)
    dtr.to_parquet(config.DEP_TRAIN, index=False)
    drk.to_parquet(config.DEP_RANK, index=False)
    v3_test(drk).to_parquet(config.DATA_DIR / "advanced_features_rank_v3.parquet", index=False)
    print(f"yazıldı: {config.DEP_TRAIN.name} {dtr.shape}, {config.DEP_RANK.name} {drk.shape}, advanced_features_rank_v3.parquet")
    print("test ayları:", drk["MVT_MONTH"].value_counts().sort_index().to_dict())
    eski = config.PRC2_DIR / "data_2" / "processed"
    if (eski / "dep_train.parquet").exists():  # yalnız tanı amaçlı karşılaştırma; eski tablolar yoksa atlanır, çıktı aynı
        karsilastir("dep_train (final geçişi ↔ PRC_2)", dtr, pd.read_parquet(eski / "dep_train.parquet"))
        karsilastir("dep_rank Oca/Tem (final ↔ PRC_2; ±12 sn oynatma beklenir)", drk, pd.read_parquet(eski / "dep_rank.parquet"))


if __name__ == "__main__":
    main()

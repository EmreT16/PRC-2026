# External input producers

These scripts are copied from our two earlier code bases (`PRC_2`, `PRC_4`) and our research folder; only comments,
the folder paths at the top and an automatic `pip install` were changed. They produce inputs that the pipeline reads
from `PRC2_DIR` / `PRC4_DIR` (see `src/config.py`) and are kept here for provenance. The folders they use must be given
as the environment variables `PRC2_DIR`, `PRC4_DIR` and, for `fetch_dest_weather.py`, `PRC_DIR`.

Order: the two weather scripts first, then the base tables of `PRC_2` (`src/prc2_taban/`, which merge the weather),
then `fetch_stand_coordinates_osm.py` and `feature_engineering_v3.py` (both use `external_runways.csv`; the second
reads the base tables from `PRC2_DIR/data_2/processed/`). The METAR scripts are independent. `fetch_dest_weather.py`
needs only the destination airports (`ADES_mvt`) of `PRC_DIR/data/dep_train.parquet` and `dep_rank.parquet`; any table
with that column, such as the raw challenge files, gives the same airport list.

| Script | Output (read by) | Source and licence |
|---|---|---|
| `fetch_weather.py` | `data/weather_all.parquet` in the working directory — hourly weather at the ten departure airports, 2025 to July 2026; with hourly METAR visibility from IEM merged in, it is stored as `PRC2_DIR/data/weather_all_v2.parquet` (`src/prc2_taban/weather_merge.py`) | [Open-Meteo](https://open-meteo.com/) archive API, CC BY 4.0 |
| `fetch_dest_weather.py` | `data/dest_weather.parquet` in the working directory, stored as `PRC2_DIR/data/dest_weather.parquet` — hourly weather at destination airports (same; the resulting `DEST_*` columns are left out of the model features in `src/features_v126.py` and `src/model_b.py`) | Open-Meteo, CC BY 4.0; airport coordinates from the `airportsdata` package |
| `fetch_stand_coordinates_osm.py` | `PRC4_DIR/data/features/stand_coordinates.csv`, `stand_rwy_distances.csv` — stand positions and stand-to-runway distances (`src/features_v126.py`) | OpenStreetMap (Overpass API, `aeroway=parking_position`), ODbL |
| `feature_engineering_v3.py` | `PRC4_DIR/data/features/advanced_features_train_v3.parquet` — runway dimensions and NM time differences for the 2025 rows (`src/features_v126.py`); the test side is rebuilt by `scripts/final_taban.py` | challenge data + runway table |
| `metar_indir.py`, `metar_saatlik.py` | `PRC_EK_VERI/metar_saatlik.parquet` — hourly METAR flags (`scripts/wx_ozellik.py`, `varis_wx_ozellik.py`); written to the repository root (ignored by git) and moved there | [IEM ASOS/METAR](https://mesonet.agron.iastate.edu/request/download.phtml), public domain |

Files downloaded as published, expected under `PRC_EK_VERI` (default `../extra_data/`, see `src/config.py`):

| File | Source | Read by |
|---|---|---|
| `ansperf/{all_pre_departure_delays, atc_pre_departure_delays, atfm_slot_adherence}_{2025,2026}.csv` | `https://www.eurocontrol.int/performance/data/download/csv/<name>_<year>.csv` | `scripts/gunluk_seri_ozellik.py`, `scripts/g_gunluk.py` |
| `ansperf/txout.xlsx` → `txout_10apt.parquet` | [ansperformance.eu data](https://www.ansperformance.eu/data/), "Taxi-out additional time" | `scripts/ansperf_txout.py` → `scripts/g_kalibre.py` |
| `aiu/aiu_pist_aylik.parquet` | [AIU airport dashboard](https://aiu-airport-dashboard.netlify.app/), monthly taxi-out per runway | `scripts/aiu_pist_indir.py` → `scripts/g_kalibre.py` |

The runway table `PRC4_DIR/data/features/external_runways.csv` is the OurAirports
[`runways.csv`](https://davidmegginson.github.io/ourairports-data/runways.csv) file as downloaded (public domain).

`PRC2_DIR/data_2/processed/dep_train.parquet` (training base table, normal mode) was built by `PRC_2/src/build_dataset.py`;
its modules are copied into `src/prc2_taban/` (only comments edited), and in final mode `scripts/final_taban.py` rebuilds both base
tables from the raw challenge files.

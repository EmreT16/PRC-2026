# Code map

Code comments and log files are in Turkish (glossary at the end of [METHOD.md](METHOD.md#9-glossary)). This page lists
the files used by the final run, in the order of `scripts/final_tum.sh`, and says in English what each one does. The
other files in `scripts/` are experiments and cross-validation runs that are not needed for the final file.

## Inputs (prepared before the run)

| File | What it does |
|---|---|
| `external/*.py` | Weather, stand distances, runway features and METAR flags; see [external/README.md](../external/README.md) |
| `scripts/adsb_hat.py` | Downloads the adsb.lol daily archives (`adsb_indir.py`) and keeps the points near the ten airports (`adsb_cikar.py`); `adsb_diger_kopya.py` takes a day from the other copy of the archive when one copy is broken |
| `scripts/opdi_indir.py` | Downloads OPDI flight events and keeps the ground events at the ten airports |
| `scripts/ansperf_txout.py` | Turns the ansperformance.eu monthly taxi-out file into a table for the ten airports |
| `scripts/aiu_pist_indir.py` | Reads the monthly taxi-out per runway from the AIU airport dashboard |
| `scripts/yapay_nm.py` | Finds narrow clusters of NM taxi values in the 2025 training data that come from NM off-block times set to take-off minus a fixed time; writes `kutuk/yapay_nm_kumeleri.csv`, read by `wx_ozellik.py` |

Commands used for the final test months (the 2025 ADS-B days are needed only to rebuild the training-side files):

```bash
python scripts/adsb_hat.py 2025-12-31:2026-02-28 2026-05-31:2026-07-31 sil=1   # about 3 GB per day, deleted after extraction
python scripts/opdi_indir.py                                                   # 2024-12-26 to 2026-08-08
python scripts/ansperf_txout.py
python scripts/aiu_pist_indir.py
```

## 1. Base tables (`final_tum.sh`)

| File | What it does |
|---|---|
| `scripts/final_taban.py` | Builds the base tables from the raw challenge files (uses `src/prc2_taban/`) |
| `scripts/final_hazirlik.py` | Copies the 2025 training-side files listed in `training_side_manifest.csv` to `data_final/` |
| `src/features_v126.py` | Base feature tables and the feature matrix of Model A |
| `scripts/final_b_taban.py` | Per-airport Model B of our earlier code base, kept at EDDF and for a small group of LIRF rows |

## 2. Features (`final_ozellik.sh`)

| File | What it does |
|---|---|
| `scripts/kuyruk_ozellik.py` | Traffic on the same runway and airport between the NM off-block and the take-off |
| `scripts/gecikme_ozellik.py` | Delay of the other departures at the airport in the last 15, 30 and 60 minutes |
| `scripts/gecikme_pist_ozellik.py` | The same per runway |
| `scripts/wx_ozellik.py` | METAR weather flags at take-off time |
| `scripts/varis_wx_ozellik.py` | Arrivals on the departure runway, with METAR |
| `scripts/pencere_ileri_ozellik.py` | Traffic after the take-off: the queue that builds up behind the flight |
| `scripts/park_varis_ozellik.py` | Previous and next arrival at the same stand |
| `scripts/komsu_ozellik.py` | Time to the neighbouring departures on the same runway and airport |
| `scripts/yanki_te_ozellik.py` | How often each flight, airline, destination and stand records the off-block equal to the schedule (out of fold) |
| `scripts/gunluk_seri_ozellik.py` | EUROCONTROL daily series per airport-day (ATFM slot share, pre-departure delay) |
| `scripts/pist_ozellik.py` | Active runway configuration |
| `scripts/opdi_ozellik.py` | OPDI ground events per departure |

## 3. Model A components (`final_a_nesil.sh`)

Model A is trained in three generations because the neighbour features need a prediction for every flight.

| File | What it does |
|---|---|
| `scripts/d_tam_egitim.py` | Trains the LightGBM, lock and XGBoost components on all of 2025 and predicts the test rows (definitions in `src/aday.py`) |
| `scripts/d_global_cb.py` | Trains the two CatBoost components (`src/gcb.py`) |
| `scripts/gcb_ortalama.py` | Averages CatBoost runs with different seeds |
| `scripts/g_test_harman.py` | Blends the components with the weights stored in `kutuk/` |
| `scripts/komsu_tah_ozellik.py` | Predicted taxi time of the neighbouring departures, from the previous generation (uses `komsu_ikinci.py`) |
| `scripts/komsu2_oz.py` | Second round of neighbour features, from the second generation |

## 4. ADS-B features and Model B (`final_son.sh`, `final_b8.sh`)

| File | What it does |
|---|---|
| `scripts/adsb_ozellik.py` | Finds take-off, start of movement, stand exit and stops in the ADS-B tracks of each departure |
| `scripts/park_b_ozellik.py` | Stand sequence and neighbour taxi features for Model B |
| `scripts/adsb_duzeltici.py` | ADS-B correction of Model A: training and prediction |
| `scripts/b_uretim_tum.py` | Model B parts trained on all departures (uses `b_model_cv_tum.py` and `b_model_cv.py`) |
| `scripts/b_lirf_te.py` | LIRF version of Model B |
| `scripts/b_lirf_tarih_kaymasi.py` | 24-hour shift mixture for long delays at LIRF |
| `scripts/adsb_b_duzeltici5.py` | ADS-B correction of Model B rows |

`final_son.sh` also builds earlier candidate files (`havalimani_agirlik.py` for airport weights, `gecikme_asama2.py`
for a second stage), and `final_a_varyant.sh` and `final_b8.sh` build candidates that were compared before the final
choice; the final file does not use them. The eight-airport Model B of the final file is assembled in `final_tabm.sh`
from the Model B parts made by `final_son.sh`.

## 5. Final configuration and post-processing (`final_tabm.sh`)

| File | What it does |
|---|---|
| `scripts/nn_bilesen.py` | TabM network, used in the Model A blend and in the ADS-B correction |
| `scripts/hedge_e1_degistir.py` | LIRF step that moves the corrected prediction part of the way towards the schedule-based value |
| `scripts/b_mu_nn.py`, `scripts/b_mu_nn_yerlestir.py` | TabM for the normal taxi time of Model B, and the rebuilt mixture |
| `scripts/b_lirf_mu_nn_uygula.py` | Partial TabM for the normal taxi time at LIRF |
| `scripts/g_gonderim.py` | Writes the submission file from Model A and Model B and checks it against the template |
| `scripts/g_kalibre.py` | Monthly calibration and LOBT window |
| `scripts/b_yetim_pencere.py` | Window rule for flights recovered through the arrival record |
| `scripts/g_gunluk.py` | Daily pre-departure delay matching and the LEMD / LEBL upper limit |

## Shared code (`src/`)

| File | What it does |
|---|---|
| `config.py` | Paths, test months and file names; `PRC_FINAL=1` selects the final test file |
| `validation.py` | Folds, out-of-fold target encoding, acceptance rules, metric |
| `aday.py` | Component definitions of Model A |
| `gcb.py` | Input and training of the CatBoost components |
| `adsb.py` | Helpers for the ADS-B correction: base blend, ADS-B feature tables, stand positions |
| `model_b.py` | Model B mixture (`fit_components_tum`, `combine_tum`) and the per-airport model used at EDDF |
| `features_temiz.py`, `models_v126.py` | Cleaned feature matrix (real categoricals, ids and constant columns dropped) used by Model A and Model B; model settings of our earlier base, used by `aday.py` |
| `prc2_taban/` | Base feature code from our earlier code base: traffic, weather, holidays, group statistics |

## Fixed settings

Some settings were chosen on 2025 data and are read from `kutuk/`: the blend weights (`g1_agirliklar_kqgpPeT2.json`),
the selected extra features (`ekoz_secim_final.json`), the residual spreads used by the daily matching
(`gunluk_kisit_sigma.json`) and the NM taxi clusters (`yapay_nm_kumeleri.csv`, made by `scripts/yapay_nm.py`).

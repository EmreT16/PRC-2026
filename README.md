# strong-moose: PRC Data Challenge 2026

This repository contains our solution for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predicting the taxi-out time of departures (`MVT_TIME - BLOCK_TIME`) at ten European airports (EDDF, EDDM, EGLL,
EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM). The metric is the RMSE in seconds.

The final submission is `strong-moose_final.parquet` (January, February, June and July 2026, 670,790 rows,
MD5 `5192dc4098c66bc2127f5be84e1646b6`). It is produced by `scripts/final_tum.sh`. The same code on the January + July
ranking set scores 217.48 on the public leaderboard (v378, one difference is noted under Leaderboard); the score of the
final file itself is not known before the end of the challenge.

A longer description of the method is in [docs/METHOD.md](docs/METHOD.md). Code comments and log files are in
Turkish; the method document ends with a short [glossary](docs/METHOD.md#9-glossary) of the Turkish names used in file
and variable names.

## Method in short

For about 98.6 % of departures the test file contains the Network Manager (NM) record, including the NM off-block
time `AOBT_3`. For these rows (Model A) we predict the gap between the airport off-block and `AOBT_3` and convert it
back to taxi time. The remaining rows (Model B) have no NM off-block, so the taxi time is predicted directly.

Model A is a blend of five gradient-boosted models (LightGBM, XGBoost, two CatBoost variants and a "lock" model for
off-block times recorded equal to the schedule) and a TabM neural network. A second model corrects the blend using
ADS-B ground tracks from adsb.lol and ground events from OPDI: it sees when the aircraft actually started moving and
learns the remaining error.

Model B handles the departures without an NM off-block. Many of these records are not real taxi times, so each
departure is treated as a mix of three cases:

- the off-block was written equal to the scheduled time, so the "taxi time" equals the departure delay;
- the off-block is on the wrong day, so the taxi time is about 24 hours too long;
- a normal taxi time.

LightGBM classifiers estimate the probability of the first two cases, LightGBM, CatBoost and TabM predict the normal
taxi time, and the prediction is the probability-weighted mix. The 24-hour classifier is trained on the 22,470
departures without NM data; the other parts are trained on all 2025 departures with the NM columns removed
(2.1 million rows). Model B is used in this form at eight airports. Frankfurt (EDDF) keeps a simpler per-airport
model, and Rome (LIRF), where the first two cases are most frequent, has its own version. Model B rows with an ADS-B track
get their own correction model, trained on out-of-fold Model B predictions, at all airports except LIRF and LTFM.

After the models, a few fixed steps are applied:

1. monthly calibration to the published EUROCONTROL taxi-out averages per airport and runway,
2. keeping the predicted off-block within 3606 s of `LOBT`, the last NM off-block time (true for every departure
   with an NM record in 2025),
3. for no-NM departures at LIRF and LTFM whose flight plan can be found through the arrival record, keeping the
   prediction outside that same window (the NM record is attached only when the off-block is inside it, so for these
   flights the true value lies outside),
4. adjusting the Model A predictions of each airport-day (except LIRF) towards the published EUROCONTROL daily
   pre-departure delay total (at most 300 s per flight),
5. a 3604 s upper limit at LEMD and LEBL (never exceeded in 2025).

Changes were first tested on 2025 data: six folds of two held-out months for Model A, four quarters for Model B
(measured on the departures without NM data). When more than one version passed, the leaderboard score decided which
one was kept, and some changes that passed on 2025 were dropped after a worse leaderboard score. In a few cases where
the 2025 test was neutral, the leaderboard decided alone. Both are listed in
[docs/METHOD.md](docs/METHOD.md#7-validation).

## Leaderboard

Public leaderboard (January + July 2026) for the submissions that improved our best score.

| Submission | Date | Change | RMSE (s) |
|---|---|---|---:|
| v145 | 29 Sep | Model A blend of five gradient-boosted models | 266.75 |
| v146-v149 | 29 Sep | Huber loss, more CatBoost iterations, seed averaging, depthwise CatBoost | 265.43 |
| v151-v152 | 29 Sep | Model B trained on all departures (6 airports) | 264.24 |
| v161 | 30 Sep | ADS-B correction of Model A | 247.47 |
| v163-v164 | 30 Sep | ADS-B correction at all airports, larger model | 245.48 |
| v170 | 30 Sep | stand positions and recording habits in the correction, LIRF schedule echo | 239.30 |
| v184 | 1 Oct | queue, delay index, arrival, METAR, forward-window and stand features | 231.19 |
| v188-v193 | 1 Oct | taxi start and stand exit from the tracks, thinned tracks, XGBoost 1,500 trees | 230.37 |
| v199-v200 | 2-3 Oct | ADS-B correction trained on 12 months | 229.68 |
| v202-v208 | 3-4 Oct | neighbour departures, schedule-echo history, EUROCONTROL daily series | 227.68 |
| v212-v224 | 5 Oct | monthly EUROCONTROL calibration (airport and runway), LOBT window, runway configuration features, more seeds | 227.15 |
| v239-v247 | 6-7 Oct | ADS-B correction for Model B rows, without NM inputs, 12 months | 225.80 |
| v255-v259 | 7 Oct | OPDI ground events, Model B correction scale 1.25 | 224.04 |
| v289-v297 | 7-8 Oct | stand sequence and neighbour taxi features, new Model B | 223.40 |
| v316-v328 | 8-9 Oct | Model B with LightGBM + CatBoost, 3 seeds; simpler Model A correction | 222.73 |
| v334 | 9 Oct | Model B mixture model at 8 airports | 221.72 |
| v345-v353 | 10 Oct | TabM in the base blend, Model B and the correction; planned times in the correction | 218.54 |
| v357-v365 | 11 Oct | 5-seed averages, TabM weight in the correction | 218.31 |
| v368 | 11 Oct | daily pre-departure delay matching | 218.04 |
| v374-v378 | 11 Oct | recovered flight plan window, final configuration | 217.48 |

v378 uses the same code and settings as the final file, with one exception: the LIRF 24-hour shift mixture is in the
final file but not in the leaderboard submissions after v320 (see [docs/METHOD.md](docs/METHOD.md#7-validation)).
The numbering continues from our earlier code bases
(v1-v144, best score 267.86); this repository starts at v145. Version numbers count the files built here; 66 of them
were uploaded, and these are listed with their scores in `kutuk/skor_defteri.csv`.

## Data

| Source | Use | Licence |
|---|---|---|
| Challenge data | training and test | provided by the organisers, not included here |
| [adsb.lol](https://github.com/adsblol) daily archives | ADS-B correction | ODbL 1.0 |
| [OPDI](https://www.opdi.aero/) flight events | ADS-B correction | open data |
| [Open-Meteo](https://open-meteo.com/) hourly weather | features | CC BY 4.0 |
| [airportsdata](https://github.com/mborsetti/airportsdata) | destination airport coordinates for the weather | MIT |
| [IEM](https://mesonet.agron.iastate.edu/) METAR reports | weather flags | public domain |
| [OurAirports](https://github.com/davidmegginson/ourairports-data), OpenStreetMap | runways, stand distances | public domain, ODbL |
| [EUROCONTROL](https://www.eurocontrol.int/performance/data/download/csv/) daily airport indicators | features, daily matching | EUROCONTROL open data |
| [ansperformance.eu](https://www.ansperformance.eu/data/) monthly taxi-out, [AIU dashboard](https://aiu-airport-dashboard.netlify.app/) | monthly calibration | EUROCONTROL open data |

The test file gives the actual take-off time and the whole day of traffic, so the task is an estimate made after the
day of operation, not a forecast. In this setting we use open data published for the same days: ADS-B positions,
OPDI events and the EUROCONTROL daily and monthly indicators. Two of these indicators, the monthly taxi-out averages
and the daily pre-departure delay totals, are published aggregates of what we predict; they enter only as airport-month
and airport-day values. All of it is open data and is declared above, as the challenge rules require.

OpenSky state vectors and the OpenSky Trino database were not used. OPDI events are built by EUROCONTROL from OpenSky
ADS-B data; we used only the published OPDI files. Tables derived from adsb.lol and OpenStreetMap are not
redistributed here. OpenStreetMap data © OpenStreetMap contributors.

## How to run

We used Python 3.13 with LightGBM 4.6, CatBoost 1.2.10, XGBoost 3.4.1, PyTorch 2.7.1 (CUDA) and `tabm` 0.0.3 (exact
versions in `requirements.txt`; the CUDA build of PyTorch comes from
`--index-url https://download.pytorch.org/whl/cu118`), on Windows 11 with Git Bash and an RTX 3050 Ti laptop GPU (4 GB).

The inputs are:

- the challenge files from the organisers' bucket,
- weather, stand distances and runway features, produced by the scripts in `external/` (see
  [external/README.md](external/README.md)),
- ADS-B points (`scripts/adsb_hat.py`) and OPDI events (`scripts/opdi_indir.py`); the download commands are in
  [docs/CODE_MAP.md](docs/CODE_MAP.md#inputs-prepared-before-the-run),
- the EUROCONTROL monthly and daily files and the AIU runway table (download links in `external/README.md`),
- the 2025 cross-validation predictions and 2025 ADS-B features made by the training runs. They are 2.8 GB and are not
  included, but every file is listed with its size, MD5 and the script that makes it in
  [docs/training_side_manifest.csv](docs/training_side_manifest.csv). No single script rebuilds them; the command
  lines are in the `kutuk/` logs and the batch scripts (`scripts/kuyruk_*.sh`, `scripts/zincir_*.sh`). We can share
  them with the organisers on request.

The scripts expect this layout. `PRC2_DIR` and `PRC4_DIR` are input data folders, named after our two earlier code
bases; the final run reads only data from them, no code. They, `PRC_EK_VERI` and `PRC_ADSB_KOK` can be set as
environment variables (defaults `../../PRC_2`, `../../PRC_4`, `../extra_data` and `../PRC_adsb`); the folders are set
in `src/config.py` and, for the ADS-B data, in the ADS-B scripts.

```
PRC2_DIR/data/               training_2025-*.parquet, weather_all_v2.parquet, dest_weather.parquet
PRC4_DIR/data/features/      external_runways.csv, stand_rwy_distances.csv, advanced_features_train_v3.parquet
../data/                     final_ranking.parquet, final_submitting.parquet and the same training_2025-*.parquet
PRC_EK_VERI/                 ansperf/, aiu/, metar_saatlik.parquet (EUROCONTROL, AIU and METAR files)
PRC_ADSB_KOK/                noktalar/ (ADS-B points), opdi/ (OPDI events)
data/                        training-side files of docs/training_side_manifest.csv
```

Then:

```bash
bash scripts/final_tum.sh > kutuk/final_tum.log 2>&1   # about 20 hours; result in gonderim_final/strong-moose_final.parquet
python tests/test_validation.py                        # unit tests: metric, acceptance rules, folds, target encoding
```

The run writes its logs to `kutuk/` and adds rows to the ledgers there, so it is best run in a copy of the
repository. A full run from scratch on 11-12 October gave a file that matches the submitted one up to an RMS difference of 7 s
(correlation 0.99993). LightGBM and XGBoost repeat exactly; CatBoost on the GPU does not give bit-identical results
between runs.

## Repository structure

```
src/        configuration, validation folds, feature matrix, Model A and Model B code
scripts/    final pipeline (final_*.sh), feature scripts (*_ozellik.py), ADS-B and OPDI (adsb_*, opdi_*),
            Model B (b_*), submission and post-processing (g_*), and the scripts of our experiments
kutuk/      logs of all runs and submissions
external/   scripts that produce inputs kept outside the repository
docs/       method description, code map, list of training-side files
tests/      unit tests
```

[docs/CODE_MAP.md](docs/CODE_MAP.md) says in English what each file of the final run does. The parts most likely to be
useful on their own are the three-case model for records without NM data (`src/model_b.py`), the extraction of taxi
start and stand exit from ADS-B tracks (`scripts/adsb_ozellik.py`), the monthly calibration (`scripts/g_kalibre.py`)
and the daily delay matching (`scripts/g_gunluk.py`). The test period and most input paths are set in
`src/config.py`. Some comments name files of our earlier code bases that parts of the code were ported from; those
files are not part of this repository.

## Originality

The code in this repository was written by our team. Public repositories of other teams and the challenge Discord
were read for ideas; every idea we used was re-implemented and tested on our own data, and no code from other teams
was copied. TabM is by Gorishniy et al. (ICLR 2025, Apache-2.0).

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).

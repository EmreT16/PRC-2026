# Method

This document describes what the final pipeline (`scripts/final_tum.sh`) does. File names refer to this repository.

## 1. Two kinds of rows

The target is the taxi-out time `MVT_TIME - BLOCK_TIME`. In the test files the take-off time `MVT_TIME` is given and
only `BLOCK_TIME` is hidden.

For about 98.6 % of the test departures (98.9 % in 2025) the Network Manager (NM) record is attached and contains `AOBT_3`, the NM actual
off-block time. For these rows (Model A) the model predicts the offset `BLOCK_TIME - AOBT_3`, and the taxi time is
`max(60, MVT_TIME - AOBT_3 - offset)`. Predicting the offset is easier than predicting the taxi time directly,
because `AOBT_3` is usually close to the real off-block.

The other departures (Model B) have no NM off-block. Their taxi time is predicted directly. These rows have
a large share of the error, because the recorded airport off-block (`BLOCK_TIME`) is often not a real event: it is
written equal to the scheduled time (then the taxi time equals the departure delay `MVT_TIME - SCHED_TIME`), or it
is on the wrong day (then the taxi time is about 24 hours too long).

## 2. Features for Model A

The feature tables are built by the `scripts/*_ozellik.py` scripts and joined in `src/features_v126.py`. From the
challenge data they use only columns that are also given in the test file. A departure's own `BLOCK_TIME` is never an
input; 2025 taxi times enter only through out-of-fold encodings computed on other months (target encodings,
schedule-echo rates and the runway-configuration encoding `PC_CFG_TE`). Weather and EUROCONTROL features come from
the open data listed in the README.

- NM time differences: take-off minus `AOBT_3`, minus the schedule and minus the NM off-block times EOBT (estimate
  updated by messages), IOBT (initial plan) and LOBT (last computed value).
- Traffic and queue: departures and arrivals on the same runway and airport between off-block and take-off, the number
  of aircraft ahead, gaps to the previous and next departure, runway throughput.
- Delay level around the flight: median delay against EOBT of the other departures in the last 15 and 30 minutes
  (per airport and per runway) and median delay against the schedule in the last 60 minutes (per airport).
- Arrivals on the departure runway and weather: Open-Meteo hourly weather and METAR flags (snow, freezing
  precipitation, fog, thunderstorm, visibility, wind).
- Traffic after the take-off. The whole day is in the test file, so the queue that builds up behind a flight can be
  used as information about its own wait.
- Stand turn-around: the previous and next arrival at the same stand.
- Neighbour departures: their spacing and their predicted taxi from an earlier model generation, how often the
  airline and the stand record the off-block equal to the schedule, the EUROCONTROL daily delay series and the active
  runway configuration.
- Out-of-fold target encodings of stand, runway, airline, route and hour.

## 3. Model A blend

| Component | Model | Weight |
|---|---|---:|
| `xgb_P` | XGBoost, 1,500 trees | 0.061 |
| `gcb_P4` | CatBoost, 12,000 iterations on the folds, 10,000 in the final fit | 0.122 |
| `e1h3_P6` | lock model: probability that the off-block equals the schedule, mixed with a LightGBM for the other rows | 0.133 |
| `lgbh3_P6` | LightGBM with Huber loss | 0.074 |
| `dw_P5` | CatBoost with depthwise trees | 0.210 |
| `tabm_T2` | TabM network (k = 8, Huber loss, 12 epochs, average of 5 seeds) | 0.400 |

The weights are stored in `kutuk/g1_agirliklar_kqgpPeT2.json`, and training uses all of 2025. The TabM share (0.4)
was chosen on the out-of-fold predictions of all six folds (each fold holds out two months). The relative weights of
the five tree models were chosen on 29 September on two folds (January + July and February + August 2025); weights
re-selected on all six folds were better on 2025 but worse on the leaderboard (v218 against v222), so the older
weights were kept.

The neighbour-prediction features need a prediction for every flight, so Model A is trained in three generations
(`scripts/final_a_nesil.sh`): the predictions of one generation become features of the next.

## 4. ADS-B correction

adsb.lol publishes daily archives of ADS-B positions. `scripts/adsb_hat.py` downloads them and keeps the points near
the ten airports. `scripts/adsb_ozellik.py` finds, for each flight, the take-off, the moment it started moving, the
stand and the stops on the way. Flights are matched by call sign and take-off time; for rows without NM data the
flight number is used instead. OPDI ground events (`scripts/opdi_ozellik.py`) cover some flights that adsb.lol only
sees on the runway.

A second model (`scripts/adsb_duzeltici.py`) learns the error of the Model A blend from these observations. On the
2025 training rows it uses the out-of-fold blend predictions, made by models that did not see that month. It is a
LightGBM model and a TabM network (weights 0.1 and 0.9), each averaged over 5 seeds. Its inputs are the ADS-B times
relative to take-off, how each airline and stand usually records the off-block compared to the moment of movement, the
OPDI events, neighbour flights and the planned times. Receiver coverage changed between 2025 and 2026 at several
airports, so the model is also trained on thinned copies of the 2025 tracks (10 % and 30 % of the points).

At LIRF many off-block times are recorded equal to the schedule. There the corrected prediction is moved part of the
way (0.25 times the lock model's probability of a schedule-equal off-block) towards the schedule-based value.

## 5. Model B

With `d = MVT_TIME - SCHED_TIME` (the departure delay), Model B predicts

`taxi = p24 * d24 + (1 - p24) * (p_sched * max(60, d) + (1 - p_sched) * mu)`

- `p24` is the probability of a 24-hour shift (LightGBM with three classes: normal, schedule-equal and 24-hour shift;
  trained on the departures without NM data), and `d24` the taxi time in that case (`d` when the delay is already
  above 40,000 s, otherwise 86,400 s plus the median of taxi time minus 86,400 s over the shifted records of 2025).
- `p_sched` is the probability that the off-block was written equal to the schedule (LightGBM, trained on all
  departures).
- `mu` is the normal taxi time: LightGBM and CatBoost (3 seeds each) blended 0.4 with TabM (5 seeds) 0.6, trained on
  all departures.

`p_sched` and `mu` are trained on all 2025 departures with the NM columns removed, which gives 2.1 million training
rows instead of 22,470. The features are the Model A base features without NM columns, plus the stand sequence and
neighbour taxi features, neighbour spacing, the EUROCONTROL daily series, the runway configuration and the
schedule-echo history. This model is used at eight airports (EDDM, EGLL, LEBL, LEMD, LFPG, LTFM, EHAM, LSZH). At EDDF
a simpler per-airport LightGBM trained on the no-NM rows was equally good on our validation and is kept. Before the
ADS-B correction, Model B predictions are capped at 5,000 s at these airports and at 3,500 s at EDDF.

LIRF has its own version of the three-case model; its classifier also uses how often each airline and stand produced
schedule-equal records in other months. For long delays that cross midnight a 24-hour shift is mixed in with a
probability estimated on 2025.

Model B rows with an ADS-B track get their own correction model at all airports except LIRF and LTFM
(`scripts/adsb_b_duzeltici5.py`). It is built like the Model A correction but trained on out-of-fold Model B
predictions, and its output is scaled by 1.25, a factor chosen on 2025 data.

## 6. Post-processing

These steps come after the models (`scripts/g_gonderim.py`, `g_kalibre.py`, `b_yetim_pencere.py`, `g_gunluk.py`).

1. Monthly calibration. ansperformance.eu publishes the average taxi-out per airport and month, and the AIU dashboard
   per runway. After removing the definition difference measured on 2025, each airport-month is shifted towards the
   published average when the difference is larger than the usual spread of that gap (one standard deviation
   for airports, three for runways).
2. LOBT window. In 2025 every departure with an NM record (2,062,577 rows) had `|BLOCK_TIME - LOBT| <= 3606 s`, so predictions are kept
   inside `MVT_TIME - LOBT ± 3606 s`.
3. Recovered flight plans. Some departures at LIRF and LTFM have no NM record, but the same flight appears as an
   arrival at another of the ten airports with its NM record. The organisers attach the NM record only when the
   off-block is inside the LOBT window above, so for these flights the real taxi time is outside the window.
   Predictions inside it are moved to the nearest edge. On 2025 this moved 73 predictions; 66 of them improved.
4. Daily delay matching. EUROCONTROL publishes, for each airport and day, the sum over departures of
   `max(0, off-block - schedule)` (`all_pre_departure_delays`; on 2025 it matches our data with a median ratio of
   0.9999). For each airport-day we adjust the Model A predictions so that the expected sum of their implied delays
   matches the published value. Each prediction moves in proportion to its uncertainty, by at most 300 s. LIRF is left out because its
   records follow a different pattern. On January, February, June and July 2025 this lowered the Model A RMSE from
   188.10 to 187.36 s. In the challenge files the last hours of the last day of each month are missing (most at
   LTFM), so on those days the published total also counts flights that are not in the file. Leaving those days out
   of the matching made no difference on 2025 (187.37 s), so they are kept.
5. At LEMD and LEBL no taxi time in 2025 was above 3604 s, so this is used as an upper limit.

## 7. Validation

Model changes were first tested on 2025 data. For Model A we used six folds, each holding out two months
(January + July, February + August, June + December, March + November, April + October, May + September). The ADS-B
correction was validated on January and July 2025 and checked again on February and June 2025, since the final also
covers those months. Model B was validated on four quarters, measured on the departures without NM data. A change
was kept when it improved the held-out months in a consistent way, for example both months, or at least three of four
quarters. When more than one version passed this test, the public leaderboard score decided between them.

Some changes that passed on 2025 were dropped because they were worse on the leaderboard (v194, v198, v218,
v301/v305). In a few cases the leaderboard decided alone, because the 2025 test was neutral or was not run for that
change. Kept this way were the runway configuration features, the runway layer of the monthly calibration (v224
against v226) and taking the LIRF lock probability from the final lock model (v351 against v349). A temperature
feature and three global scales of the Model A ADS-B correction (1.15, 0.9 and 0.883) were tried and not kept, so the
final correction is not scaled. An early rule
(v153) that changed only three LIRF rows was worse on the leaderboard and was removed; the 24-hour mixture in the final
file, fitted on 2025, moves the same rows in the same direction again. No prediction was set from the leaderboard
score of single rows. All submissions and their scores are in `kutuk/skor_defteri.csv`.

## 8. Reproducibility

`scripts/final_tum.sh` rebuilds the final file from the challenge data, the external inputs and the 2025 training-side
predictions listed in `docs/training_side_manifest.csv`. It takes about 20 hours on our laptop.

LightGBM, XGBoost and the post-processing give identical results on every run. CatBoost on the GPU does not: the order
of floating-point sums changes between runs, so its first-generation predictions differ by 23-29 s RMS
(`kutuk/catboost_gpu_tekrar.log`), and through the neighbour features this reaches the later models. A complete run
from scratch on 11-12 October produced a file with an RMS difference of 6.99 s from the submitted one (correlation
0.99993). `scripts/final_karsilastir.py` compares two final files. After that run the code of the final pipeline was
changed only in file paths, input checks, guarded diagnostics and comments; we checked this by comparing the syntax
trees of both versions.

## 9. Glossary

| Turkish | English | Turkish | English |
|---|---|---|---|
| taban | base | düzeltici (`duz`) | corrector |
| harman | blend | tohum | seed |
| özellik (`oz`) | feature | tahmin | prediction |
| eğitim | training | doğrulama | validation |
| gönderim | submission | kütük (`kutuk`) | log |
| havalimanı | airport | pist | runway |
| park (yeri) | stand | kalkış / varış | departure / arrival |
| gecikme | delay | kuyruk | queue |
| komşu | neighbour | yankı | schedule echo |
| kilit | lock (off-block = schedule) | yetim | orphan (no NM record) |
| günlük | daily | aylık | monthly |
| zincir | batch script | aday | candidate |

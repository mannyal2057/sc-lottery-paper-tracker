# WEATHER-V1

Added September 15, 2026. Run `python weather_experiment.py` after `python paper_track.py` and `python three_pick_review.py`. Existing Python dependencies suffice. No account, weather API key, or paid subscription is required for the public endpoints used here.

The separate `weather/` folder contains the protocol, public source snapshots, three-pick forecasts, and current comparison. Forecasts are created once and not overwritten. The original model remains unchanged. Daily checks compare both against the same results verified by the original tracker. Missing weather produces an issue, not invented inputs.

## Inputs and limitations

SCEL identifies Studio-on-Main, Columbia, as the drawing location: https://www.sceducationlottery.com/Games/HowtoPlay and https://sceducationlottery.com/StudioOnMain . We use an approximate downtown outdoor grid point, not indoor measurements or an exact weather station at the studio. No historical site continuity before 2020 is assumed.

Open-Meteo supplies temperature (Celsius), relative humidity (%) and surface pressure (hPa): https://open-meteo.com/en/docs/historical-weather-api . Archived ERA5 data are reanalysis estimates; they can incorporate observations unavailable at the time of a historical drawing. Thus the retrospective evaluation is an exploratory association test, not an honest simulation of as-issued forecasts. Live predictions instead use forecast snapshots saved before the draw cutoff. The difference between reanalysis and forecasts is a limitation; indoor climate control may disconnect both from the mechanism.

## Fixed comparison

Separate models for midday and evening. Each digit position has a regularized logistic regression. Both models use annual seasonal sine/cosine and weekday; only the weather model adds the three weather variables. Standardization fits on training rows only. Settings are fixed in protocol.json. Models combine digit probabilities under an independence assumption and choose the three highest-probability numbers. The calendar model is the matched control; the original ensemble is also shown in live reports.

Training starts in 2020. Historical test folds are 2023, 2024 and 2025, each trained only on earlier years. No target outcome enters the features. Recent live training stops at least seven days before the target to allow weather archive availability. Version changes require a new experiment; do not modify this version to explain misses.

| Historical test | Drawings | Weather exact hits | Calendar exact hits | Weather log loss | Calendar log loss |
|---|---:|---:|---:|---:|---:|
| Midday | 936 | 2 | 2 | 6.939077 | 6.931212 |
| Evening | 1,096 | 6 | 6 | 6.919690 | 6.912529 |

Lower log loss is better. Uniform probabilities yield 6.907755. Weather has not demonstrated an advantage in this test. Aggregate equal hit counts do not mean that the same dates hit. Full yearly results are in backtest.json. No settings were chosen by selecting the best of these folds.

## Operation and portability

The existing 9 a.m./3 p.m. Eastern automation includes this script. Weekly reviews will include the separate experiment. The original study's stopping rule remains in force; the weather study may have fewer observations at that point.

For an existing portable SC_PICK3 package, merge this add-on's weather_experiment.py and weather folder into its root, then run the three commands above. Saved reports work offline; updates require public internet. Automations do not transfer to another account. Preserve protocol.json and source records. Unit checks: `python -m pytest tests/test_weather.py -q`.

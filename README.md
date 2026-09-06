# Chandigarh Next-Day AQI Predictor

Predicts **next-day Air Quality Index (AQI)** for Chandigarh, India, during peak stubble-burning season (Oct–Nov), using satellite fire data, trace-gas columns, boundary-layer height, and wind — compared across **XGBoost** and an **LSTM**.

**Study period:** Oct–Nov 2021, 2022, 2023
**City:** Chandigarh UT (30.73°N, 76.79°E) — CPCB stations: Sector 22, 25, 53
**Notebook:** `chandigarh_aqi_predictor.ipynb` (built for Google Colab, GPU runtime)

## Why

Stubble burning in upwind Punjab/Haryana farmland drives severe AQI spikes in Chandigarh each autumn. This notebook tests whether adding **fire radiative power (FRP)** and, novelly, **boundary layer height (BLH)** — a proxy for how trapped smoke is near the surface — improves next-day AQI forecasts beyond trace gases and wind alone.

## Data sources

| Source | Dataset | Access |
|---|---|---|
| NASA FIRMS | VIIRS S-NPP fire FRP (375 m) | firms.modaps.eosdis.nasa.gov |
| Copernicus / Google Earth Engine | Sentinel-5P TROPOMI NO₂ + CO | earthengine.google.com |
| ECMWF / GEE | ERA5-Land 10 m wind (u/v) | earthengine.google.com |
| NASA / GEE | MERRA-2 boundary layer height | earthengine.google.com |
| CPCB / UrbanEmissions | Daily AQI, Chandigarh | urbanemissions.info / cpcbccr.com |

## Pipeline (notebook sections)

0. Install dependencies (`earthengine-api`, `geemap`, `xgboost`, `shap`, `torch`, etc.)
1. Authenticate to Google Earth Engine
2. Configuration — city coordinates, upwind fire bounding box, 3 seasons, FIRMS key, CPCB station coordinates
3. Load CPCB AQI data (from CSV, or synthetic fallback calibrated to published annual means)
4. Fetch VIIRS fire detections (FIRMS API), computing distance-weighted FRP relative to Chandigarh
5. Fetch Sentinel-5P TROPOMI NO₂ / CO daily means via GEE
6. Fetch ERA5-Land wind speed/direction via GEE
7. Fetch MERRA-2 boundary layer height via GEE (hydrostatic approximation from `PBLTOP`)
8. Merge all sources into a master feature table; add lag (t-1, t-2) and rolling-mean features
9. Exploratory 3-season comparison plots + correlation heatmap
10. Temporal train/test split: **train on 2021+2022, test on 2023** (out-of-sample year, not a random split)
11. Train **XGBoost** regressor
12. Train **LSTM** (PyTorch, 7-day sliding window, early stopping on validation loss)
13. Head-to-head comparison: time series, scatter plots, MAE/RMSE/R²
14. SHAP feature importance for XGBoost
15. AQI-category classification report (Good/Satisfactory/.../Severe) + confusion matrix for the better model
16. BLH deep-dive: shows that the same fire intensity produces worse AQI when BLH is shallow
17. Interactive Folium map of fire hotspots + CPCB stations
18. Save a JSON summary of the run (metrics, features, top SHAP drivers)
19. Suggested paper outline and target journals for write-up

## Key engineered features

- Distance-decayed fire metrics: `total_frp`, `frp_weighted`, `fire_count`, `max_frp`
- Rolling fire/AQI/BLH means (3-day, 7-day)
- `frp_blh_ratio` — the paper's proposed novel interaction feature
- Lagged AQI, FRP, NO₂, CO, BLH (t-1, t-2)
- Calendar features: day of year, week, month, year

## Requirements

- Google Colab with a **T4 GPU runtime** (`Runtime → Change runtime type → T4 GPU`)
- A Google Earth Engine account and project (free signup)
- A NASA FIRMS API key (free, emailed within minutes)
- Optionally, a CPCB/UrbanEmissions AQI CSV (otherwise synthetic data is used)

## Before you run

Update these placeholders in the Configuration and FIRMS cells:
- `ee.Initialize(project='...')` → your own GEE project ID
- `FIRMS_API_KEY` → your own NASA FIRMS key

> ⚠️ **Security note:** the current notebook has a real-looking FIRMS API key hardcoded in Section 4. Treat it as compromised — replace it with your own key and avoid committing personal API keys to notebooks you share.

## Known issues in the current notebook

- **Section 4 (`fetch_and_agg`)**: `pd.read_csv()` is called with no arguments — it needs the constructed `url` passed in, e.g. `pd.read_csv(url)`. This cell will error as written.
- **Section 17 (Folium map)**: fire hotspots are plotted from `np.random` synthetic points, not the real FIRMS detections fetched earlier — replace with actual FIRMS lat/lon once Section 4 is fixed.
- Several output cells were truncated when exported and may have additional plotting/formatting code not captured here.

## Outputs

Running the full notebook produces:
- `correlation_heatmap.png`, `shap_importance.png`, `lstm_curve.png`
- `xgb_chandigarh.json` (saved XGBoost model), `lstm_chandigarh.pt` (saved LSTM weights)
- A JSON run summary with MAE/RMSE/R² for both models and top SHAP features
- An interactive Folium map (rendered inline)

## Models compared

| Model | Approach |
|---|---|
| XGBoost | Gradient-boosted trees, 500 estimators, early stopping on 2023 test set |
| LSTM | 2-layer PyTorch LSTM over 7-day feature windows, trained 150 epochs with early stopping on validation loss |

Both are evaluated with MAE, RMSE, R², and accuracy at the Indian AQI category level (Good → Severe).

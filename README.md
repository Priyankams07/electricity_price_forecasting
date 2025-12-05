# Electricity Price Forecasting using ANN

**Project:** Hourly day-ahead electricity price forecasting using a Feedforward ANN (MLP).  
**Language:** Python | **Libraries:** pandas, numpy, scikit-learn, tensorflow, matplotlib, joblib.

---

## Quick overview
This repo contains a modular Python project that:
- Loads ENTSO-E / similar hourly price CSV data,
- Creates lag and time features,
- Trains a feedforward neural network (ANN),
- Saves scalers and model artifacts,
- Produces forecasts and prediction plots.

---

## Repository structure
electricity_price_forecasting/
├── data/
│ └── electricity_dah_prices.csv
├── models/
│ └── ann_model.keras (saved model)
│ └── sx.save (feature scaler)
│ └── sy.save (target scaler)
├── results/
│ └── predictions.png
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── train.py
│ └── predict.py
├── requirements.txt
└── README.md

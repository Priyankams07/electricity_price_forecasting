# src/predict.py
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.config import LAGS

def forecast_next_hour():
    try:
        df = load_data()
        df = create_features(df, lags=LAGS)
        X_df = df.drop("price", axis=1)
        y_df = df[["price"]]

        # load scalers saved during training
        sx = joblib.load("models/sx.save")
        sy = joblib.load("models/sy.save")

        X_scaled = sx.transform(X_df.values)

        # load native keras model (recommended)
        model = load_model("models/ann_model.keras", compile=False)

        last_input = X_scaled[-1].reshape(1, -1)
        y_pred_scaled = model.predict(last_input)
        y_pred = sy.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]

        print(f"Scaled prediction: {y_pred_scaled.flatten()[0]:.6f}")
        print(f"Predicted price (original units): {y_pred:.6f}")
        return y_pred

    except Exception as e:
        print("Error during prediction:", e, file=sys.stderr)
        raise

if __name__ == "__main__":
    forecast_next_hour()

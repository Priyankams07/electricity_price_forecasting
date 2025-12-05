# src/train.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib

from src.data_loader import load_data
from src.feature_engineering import create_features
from src.model import build_model
from src.config import MODEL_PATH, TEST_SIZE, LAGS, EPOCHS, BATCH_SIZE

# ensure models/results dirs exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def train():
    # Load and prepare data
    df = load_data()
    df = create_features(df, LAGS)

    X_df = df.drop("price", axis=1)
    y_df = df[["price"]]

    # Scalers
    sx = MinMaxScaler()
    sy = MinMaxScaler()

    X = sx.fit_transform(X_df.values)
    y = sy.fit_transform(y_df.values)

    # Train-test split (time-series style, no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )

    # Build and train model
    model = build_model(X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Predictions (scaled)
    y_pred_scaled = model.predict(X_test)

    # Inverse-transform to original units
    y_test_orig = sy.inverse_transform(y_test)
    y_pred_orig = sy.inverse_transform(y_pred_scaled)

    real_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    print("TEST RMSE (original units):", real_rmse)

    # Save scalers
    joblib.dump(sx, "models/sx.save")
    joblib.dump(sy, "models/sy.save")

    # Save model in native Keras format and legacy HDF5 (optional)
    model.save("models/ann_model.keras")      # recommended native format
    model.save(MODEL_PATH)                    # existing h5 for compatibility
    print("Scalers and models saved to models/")

    # Save prediction plot (original units)
    plt.figure(figsize=(12,5))
    # plot original-unit arrays (flatten them)
    plt.plot(y_test_orig.flatten(), label="Actual")
    plt.plot(y_pred_orig.flatten(), label="Predicted")
    plt.legend()
    plt.title("Electricity Price Forecasting (original units)")
    plt.xlabel("Test sample index")
    plt.ylabel("Price (original units)")
    plt.tight_layout()
    plt.savefig("results/predictions.png")
    plt.show()

if __name__ == "__main__":
    train()

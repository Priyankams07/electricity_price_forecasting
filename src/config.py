DATA_PATH = "data/electricity_dah_prices.csv"
TARGET_COLUMN = "france"   # use the exact header as it appears in the CSV (check file)
MODEL_PATH = "models/ann_model.h5"
TEST_SIZE = 0.2
LAGS = 24
EPOCHS = 10
BATCH_SIZE = 32

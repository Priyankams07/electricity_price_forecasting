def create_features(df, lags=24):
    # LAG features
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["price"].shift(lag)

    # TIME features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    return df.dropna()

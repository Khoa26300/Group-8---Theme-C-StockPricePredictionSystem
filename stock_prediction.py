from collections import deque
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, GRU, Input, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential

from parameters import LOOKUP_STEP, N_STEPS, SCALE


np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


DEFAULT_FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

RECURRENT_LAYERS = {
    "LSTM": LSTM,
    "GRU": GRU,
    "SimpleRNN": SimpleRNN,
}


def shuffle_in_unison(a, b):
    """Shuffle two arrays in the same order.

    This keeps features and targets aligned after we randomise the order of a
    split that was generated sequentially from a time series.
    """

    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def _normalise_column_name(column_name):
    """Convert column labels to the lowercase names used by the project."""

    cleaned = str(column_name).strip().lower().replace(" ", "_")
    if cleaned in {"adj_close", "adjclose"}:
        return "adjclose"
    return cleaned


def _standardise_dataframe(raw_df):
    """Return a copy of the dataframe with predictable lowercase columns.

    The project code always refers to columns such as `adjclose`, `open`, and
    `volume`, so we normalise whatever comes from yfinance or a local CSV into
    that shape before any other preprocessing happens.
    """

    df = raw_df.copy()
    df.columns = [_normalise_column_name(column) for column in df.columns]

    if "adjclose" not in df.columns and "close" in df.columns:
        # Some CSV exports only contain the closing price.  Reuse it as the
        # adjusted close so the rest of the pipeline still works.
        df["adjclose"] = df["close"]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = ["adjclose", "close", "high", "low", "open", "volume"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    df.index.name = "date"

    return df


def _resolve_cache_path(ticker, start_date, end_date, local_data_path=None):
    """Build a stable cache path for downloaded data."""

    if local_data_path:
        cache_path = Path(local_data_path)
        if cache_path.suffix:
            return cache_path
        safe_ticker = str(ticker).replace("/", "_").replace(".", "_")
        return cache_path / f"{safe_ticker}_{start_date}_{end_date}.csv"

    safe_ticker = str(ticker).replace("/", "_").replace(".", "_")
    return Path("data") / f"{safe_ticker}_{start_date}_{end_date}.csv"


def _apply_nan_strategy(df, nan_method):
    """Handle missing values with the strategy requested by the user."""

    if nan_method == "drop":
        return df.dropna()
    if nan_method == "ffill":
        return df.ffill().bfill()
    if nan_method == "bfill":
        return df.bfill().ffill()
    if nan_method == "interpolate":
        return df.interpolate(method="linear").ffill().bfill()
    if nan_method == "none":
        return df
    raise ValueError("nan_method must be one of: drop, ffill, bfill, interpolate, none")


def _load_raw_dataframe(ticker, start_date, end_date, load_local_data, save_local_data, local_data_path):
    """Load data from disk when possible, otherwise download it."""

    if isinstance(ticker, pd.DataFrame):
        return ticker.copy(), None

    if not isinstance(ticker, str):
        raise TypeError("ticker can be either a str or a pd.DataFrame instance")

    cache_path = _resolve_cache_path(ticker, start_date, end_date, local_data_path)
    if load_local_data and cache_path.exists():
        raw_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return raw_df, cache_path

    raw_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if save_local_data:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(cache_path)

    return raw_df, cache_path


def load_data(
    ticker,
    n_steps=50,
    scale=True,
    shuffle=True,
    lookup_step=1,
    split_by_date=True,
    test_size=0.2,
    feature_columns=None,
    start_date="2020-01-01",
    end_date="2024-07-02",
    nan_method="drop",
    save_local_data=True,
    load_local_data=True,
    local_data_path=None,
):
    """Load, clean, cache, scale and split a stock dataset.

    The returned dictionary keeps the raw cleaned dataframe, the training and
    testing tensors, the scaler objects, and the last sequence needed for the
    one-step-ahead prediction in `test.py`.
    """

    if feature_columns is None:
        feature_columns = list(DEFAULT_FEATURE_COLUMNS)
    else:
        feature_columns = [_normalise_column_name(column) for column in feature_columns]

    # The target in this project is always adjusted close.  Make sure it is
    # present in the feature list so the model can both consume it and later
    # invert the scaling back to a price.
    if "adjclose" not in feature_columns:
        feature_columns = ["adjclose", *feature_columns]

    raw_df, _ = _load_raw_dataframe(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        load_local_data=load_local_data,
        save_local_data=save_local_data,
        local_data_path=local_data_path,
    )

    if raw_df.empty:
        raise ValueError("No rows were loaded for the requested ticker and date range")

    df = _standardise_dataframe(raw_df)

    # If the caller passed a local dataframe, keep only the requested date
    # window when the index is actually datetime-like.
    if isinstance(df.index, pd.DatetimeIndex):
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

    df = df.sort_index()
    df = _apply_nan_strategy(df, nan_method)

    required_columns = {"open", "high", "low", "close", "volume", "adjclose"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns after preprocessing: {missing_columns}")

    # Keep a cleaned but unscaled copy for reporting and plotting.
    cleaned_df = df.copy()
    cleaned_df["date"] = cleaned_df.index

    working_df = df.copy()

    column_scaler = {}
    if scale:
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            working_df[column] = scaler.fit_transform(np.expand_dims(working_df[column].values, axis=1))
            column_scaler[column] = scaler

    # The future label is the adjusted close shifted forward by `lookup_step`.
    # This gives the model a supervised target while keeping the input window as
    # the previous `n_steps` rows.
    working_df["future"] = working_df["adjclose"].shift(-lookup_step)
    cleaned_df["future"] = cleaned_df["adjclose"].shift(-lookup_step)

    if len(working_df) < n_steps + lookup_step:
        raise ValueError("Not enough rows to build the requested sequence length")

    feature_matrix = working_df[feature_columns].values
    target_values = working_df["future"].values
    date_values = working_df.index.to_numpy()

    sequence_data = []
    sequences = deque(maxlen=n_steps)
    sequence_dates = deque(maxlen=n_steps)

    for row_index, (row_values, target_value) in enumerate(zip(feature_matrix, target_values)):
        sequences.append(row_values)
        sequence_dates.append(date_values[row_index])

        # Each sample is a rolling window of `n_steps` rows with one future
        # target.  We attach the last date in the window so the test split can be
        # aligned back to the original dataframe for plotting and profit logic.
        if len(sequences) == n_steps and not pd.isna(target_value):
            sequence_data.append(
                {
                    "X": np.array(sequences, dtype=np.float32),
                    "y": np.float32(target_value),
                    "date": sequence_dates[-1],
                }
            )

    if not sequence_data:
        raise ValueError("No valid sequences could be created from the dataset")

    X = np.array([sample["X"] for sample in sequence_data], dtype=np.float32)
    y = np.array([sample["y"] for sample in sequence_data], dtype=np.float32)
    dates = np.array([sample["date"] for sample in sequence_data])

    result = {
        "df": cleaned_df,
        "feature_columns": feature_columns,
        "last_sequence": np.array(working_df[feature_columns].tail(n_steps), dtype=np.float32),
        "column_scaler": column_scaler,
    }

    if split_by_date:
        train_size = int((1 - test_size) * len(X))
        train_size = max(1, min(train_size, len(X) - 1))

        result["X_train"] = X[:train_size]
        result["y_train"] = y[:train_size]
        result["X_test"] = X[train_size:]
        result["y_test"] = y[train_size:]
        result["test_dates"] = dates[train_size:]

        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        train_random_state = 314 if shuffle else None
        X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
            X,
            y,
            dates,
            test_size=test_size,
            shuffle=shuffle,
            random_state=train_random_state,
        )

        result["X_train"] = X_train
        result["y_train"] = y_train
        result["X_test"] = X_test
        result["y_test"] = y_test
        result["test_dates"] = test_dates
        result["train_dates"] = train_dates

    # Keep the test dataframe unscaled so evaluation and plots use real prices.
    test_index = pd.Index(result["test_dates"])
    result["test_df"] = cleaned_df.loc[test_index].copy()
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep="first")]

    return result


def _resolve_cell(cell):
    if isinstance(cell, str):
        cell_name = cell.strip()
        if cell_name not in RECURRENT_LAYERS:
            raise ValueError(f"Unsupported recurrent layer type: {cell}")
        return RECURRENT_LAYERS[cell_name]
    return cell


def _resolve_units(units, n_layers):
    if isinstance(units, int):
        return [units] * n_layers

    resolved_units = list(units)
    if len(resolved_units) == 1 and n_layers > 1:
        return resolved_units * n_layers
    if len(resolved_units) != n_layers:
        raise ValueError("Layer units must contain one value per layer, or a single value to reuse.")
    return resolved_units


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    """Create and compile a stacked recurrent model from flexible layer settings."""

    cell = _resolve_cell(cell)
    layer_units = _resolve_units(units, n_layers)
    model = Sequential()
    model.add(Input(shape=(sequence_length, n_features)))

    for layer_index, layer_units_count in enumerate(layer_units):
        is_first_layer = layer_index == 0
        is_last_layer = layer_index == n_layers - 1
        layer_kwargs = {"return_sequences": not is_last_layer}

        if is_first_layer:
            if bidirectional:
                model.add(Bidirectional(cell(layer_units_count, **layer_kwargs)))
            else:
                model.add(cell(layer_units_count, **layer_kwargs))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(layer_units_count, **layer_kwargs)))
            else:
                model.add(cell(layer_units_count, **layer_kwargs))

        model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


def get_final_df(model, data):
    """Construct a dataframe with the true and predicted test prices."""

    buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

    X_test = data["X_test"]
    y_test = data["y_test"]
    y_pred = model.predict(X_test)

    if hasattr(model, "predict") and SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

    test_df = data["test_df"]
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    test_df.sort_index(inplace=True)
    final_df = test_df

    final_df["buy_profit"] = list(
        map(
            buy_profit,
            final_df["adjclose"],
            final_df[f"adjclose_{LOOKUP_STEP}"],
            final_df[f"true_adjclose_{LOOKUP_STEP}"],
        )
    )
    final_df["sell_profit"] = list(
        map(
            sell_profit,
            final_df["adjclose"],
            final_df[f"adjclose_{LOOKUP_STEP}"],
            final_df[f"true_adjclose_{LOOKUP_STEP}"],
        )
    )
    return final_df


def predict(model, data):
    """Predict the next future adjusted close from the last sequence."""

    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)

    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]

    return predicted_price
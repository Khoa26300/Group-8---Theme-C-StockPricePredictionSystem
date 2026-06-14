import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import mplfinance as mpf
except ImportError:
    mpf = None

from parameters import *
from stock_prediction import create_model, load_data


def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def _validate_ohlcv_dataframe(df):
    """Validate and standardize an OHLCV dataframe for financial plots.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least Open, High, Low,
            Close and Volume columns (case-insensitive).

    Returns:
        pd.DataFrame: Copy with lowercase column names and a DatetimeIndex.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    prepared = df.copy()
    prepared.columns = [str(column).strip().lower() for column in prepared.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in prepared.columns]
    if missing:
        raise ValueError(f"Missing required columns for charting: {missing}")

    if "adjclose" not in prepared.columns and "close" in prepared.columns:
        prepared["adjclose"] = prepared["close"]

    if not isinstance(prepared.index, pd.DatetimeIndex):
        if "date" in prepared.columns:
            prepared.index = pd.to_datetime(prepared["date"], errors="coerce")
        else:
            prepared.index = pd.to_datetime(prepared.index, errors="coerce")

    prepared = prepared[~prepared.index.isna()].sort_index()
    return prepared


def _aggregate_trading_days(ohlcv_df, n_days):
    """Aggregate OHLCV rows into groups of n consecutive trading days.

    Parameters:
        ohlcv_df (pd.DataFrame): Standardized OHLCV dataframe indexed by date.
        n_days (int): Number of trading days per candle. Must be >= 1.

    Returns:
        pd.DataFrame: Aggregated OHLCV dataframe indexed by the last day in each
            n-day group.
    """

    if n_days < 1:
        raise ValueError("n_days must be >= 1")

    if n_days == 1:
        return ohlcv_df[["open", "high", "low", "close", "volume"]].copy()

    grouped_rows = []
    for start in range(0, len(ohlcv_df), n_days):
        block = ohlcv_df.iloc[start : start + n_days]
        if block.empty:
            continue

        grouped_rows.append(
            {
                "date": block.index[-1],
                "open": block["open"].iloc[0],
                "high": block["high"].max(),
                "low": block["low"].min(),
                "close": block["close"].iloc[-1],
                "volume": block["volume"].sum(),
            }
        )

    aggregated = pd.DataFrame(grouped_rows).set_index("date")
    return aggregated


def plot_candlestick_chart(df, n_days=1, title=None, style="yahoo", show_volume=True):
    """Plot stock data as a candlestick chart.

    Parameters:
        df (pd.DataFrame): Input market data with OHLCV columns.
        n_days (int): Number of trading days represented by each candlestick.
            Example: n_days=5 means one candle per 5 consecutive trading days.
        title (str|None): Optional chart title. When None, a default title is used.
        style (str): mplfinance style preset (for example: 'yahoo', 'charles').
        show_volume (bool): If True, draw a volume subplot under the price chart.

    Returns:
        None
    """

    if mpf is None:
        print("[WARN] mplfinance is not installed. Run: pip install mplfinance")
        return

    prepared = _validate_ohlcv_dataframe(df)
    ohlcv = _aggregate_trading_days(prepared, n_days=n_days)

    if title is None:
        title = f"Candlestick Chart ({n_days} Trading Day(s) per Candle)"

    # mpf.plot arguments:
    # - type='candle'     : render candlestick bodies and wicks
    # - style=style       : applies a predefined visual theme
    # - volume=show_volume: include volume bars in a separate panel
    # - mav=(3, 6)        : overlay simple moving averages for trend context
    # - datetime_format   : controls how dates are shown on the x-axis
    mpf.plot(
        ohlcv,
        type="candle",
        style=style,
        volume=show_volume,
        mav=(3, 6),
        title=title,
        ylabel="Price",
        ylabel_lower="Volume",
        datetime_format="%Y-%m-%d",
    )


def plot_moving_window_boxplot(
    df,
    window_size=20,
    step=5,
    value_column="adjclose",
    max_windows=30,
    title=None,
):
    """Plot a boxplot over a moving window of consecutive trading days.

    Parameters:
        df (pd.DataFrame): Input market dataframe.
        window_size (int): Number of consecutive trading days in each window.
        step (int): Sliding step between windows. Smaller step gives denser plots.
        value_column (str): Numeric column to summarize in each box.
        max_windows (int): Maximum number of windows drawn to avoid clutter.
        title (str|None): Optional figure title.

    Returns:
        None
    """

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")

    prepared = _validate_ohlcv_dataframe(df)
    if value_column not in prepared.columns:
        raise ValueError(f"Column '{value_column}' was not found in dataframe")

    values = prepared[value_column].astype(float)
    box_data = []
    labels = []

    for start in range(0, len(values) - window_size + 1, step):
        end = start + window_size
        window_slice = values.iloc[start:end]
        box_data.append(window_slice.values)
        labels.append(window_slice.index[-1].strftime("%Y-%m-%d"))

    if not box_data:
        raise ValueError("Not enough rows for the chosen window_size and step")

    # Keep the plot readable by only showing the most recent windows.
    if len(box_data) > max_windows:
        box_data = box_data[-max_windows:]
        labels = labels[-max_windows:]

    plt.figure(figsize=(14, 6))
    plt.boxplot(box_data, labels=labels, showfliers=False)

    if title is None:
        title = f"Moving Window Boxplot ({value_column}, window={window_size}, step={step})"

    plt.title(title)
    plt.xlabel("Window End Date")
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# Load the data with the same preprocessing settings used during training.
data = load_data(
    ticker,
    N_STEPS,
    scale=SCALE,
    split_by_date=SPLIT_BY_DATE,
    shuffle=SHUFFLE,
    lookup_step=LOOKUP_STEP,
    test_size=TEST_SIZE,
    feature_columns=FEATURE_COLUMNS,
    start_date=START_DATE,
    end_date=END_DATE,
    nan_method=NAN_METHOD,
    save_local_data=SAVE_LOCAL_DATA,
    load_local_data=LOAD_LOCAL_DATA,
    local_data_path=LOCAL_DATA_PATH,
)

# Construct the model with the exact feature count returned by preprocessing.
model = create_model(N_STEPS, len(data["feature_columns"]), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# Load the best weights from the training phase.
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# Evaluate the model on the held-out test split.
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# Build the final dataframe with true and predicted prices.
final_df = get_final_df(model, data)
# Predict the next future price after the lookup window.
future_price = predict(model, data)
# Accuracy is measured by counting how often the sign of the trade is correct.
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
# Calculate the total buy and sell profit.
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# Total profit is the sum of buy and sell profit.
total_profit = total_buy_profit + total_sell_profit
# Divide total profit by the number of test trades.
profit_per_trade = total_profit / len(final_df)
# Print the evaluation metrics.
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)
# Plot the true and predicted price graph.
plot_graph(final_df)

# Task 3 visualizations (v0.2): candlestick and moving-window boxplot.
# The parameters come from parameters.py so you can adjust them without
# changing code here.
if ENABLE_TASK3_PLOTS:
    plot_candlestick_chart(
        data["df"],
        n_days=CANDLE_N_DAYS,
        title=f"{ticker} Candlestick ({CANDLE_N_DAYS} Day Candle)",
        style="yahoo",
        show_volume=True,
    )

    plot_moving_window_boxplot(
        data["df"],
        window_size=BOXPLOT_WINDOW_SIZE,
        step=BOXPLOT_STEP,
        value_column=BOXPLOT_VALUE_COLUMN,
        max_windows=BOXPLOT_MAX_WINDOWS,
        title=f"{ticker} Moving Window Boxplot ({BOXPLOT_VALUE_COLUMN})",
    )

print(final_df.tail(10))
# Save the final dataframe to the csv-results folder.
csv_results_folder = "csv-results"
if not os.path.isdir(csv_results_folder):
    os.mkdir(csv_results_folder)
csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
final_df.to_csv(csv_filename)

# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance
# pip install mplfinance
# pip install statsmodels

import os
import random
import datetime as dt
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, Input, GRU, SimpleRNN, Bidirectional
from statsmodels.tsa.arima.model import ARIMA

# Set a fixed random seed to ensure reproducible and consistent results across runs
np.random.seed(314)
random.seed(314)


# =================================================----------------------------
# TASK 2 FUNCTIONS (FROM function.py)
# =================================================----------------------------

def shuffle_in_unison(a, b):
    """
    Shuffles two NumPy arrays simultaneously so that the row-by-row mapping
    between features (a) and labels (b) is perfectly preserved.
    """
    # Capture the internal state of the random number generator
    state = np.random.get_state()
    # Randomly shuffle the feature array in-place
    np.random.shuffle(a)
    # Restore the exact same random state to synchronize the next shuffle
    np.random.set_state(state)
    # Shuffle the target array using the exact same permutation order
    np.random.shuffle(b)


def load_data(
    ticker,
    n_steps=50,
    scale=True,
    shuffle=True,
    lookup_step=1,
    split_by_date=True,
    test_size=0.2,
    feature_columns=["adjclose", "volume", "open", "high", "low"],
    start_date="2020-01-01",  # Requirement 1a: Custom start date parameter
    end_date="2023-08-01",  # Requirement 1a: Custom end date parameter
    load_local=True,  # Requirement 1d: Toggle to load data from a local file
    store_local=True,  # Requirement 1d: Toggle to save downloaded data locally
):
    """
    Loads stock data, handles missing values, normalizes features, caches files,
    and splits the dataset into training and testing sets based on user preferences.
    """
    # Define a clean, dynamic filename based on the asset ticker and date range
    local_filename = f"data/{ticker}_{start_date}_to_{end_date}.csv"

    # Requirement 1d: Check if local loading is enabled and the file exists on disk
    if load_local and os.path.exists(local_filename):
        print(f"[INFO] Loading dataset locally from: {local_filename}")
        # Read the local CSV file and parse its index column as date objects
        df = pd.read_csv(local_filename, index_col=0, parse_dates=True)
    else:
        print(f"[INFO] Fetching data live from Yahoo Finance for: {ticker}")
        # Requirement 1a & 1d: Download live data via the API using explicit date boundaries
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)

        # Standardize all column headers to lowercase strings to eliminate case mismatch bugs
        df.columns = df.columns.str.lower()

        # Clone the standard "close" price array into an "adjclose" column to support legacy logic
        df["adjclose"] = df["close"]

        # Requirement 1d: If local saving is enabled, ensure directory exists and write out the file
        if store_local:
            if not os.path.isdir("data"):
                os.mkdir("data")
            # Export the pristine, unscaled DataFrame out to a static CSV file
            df.to_csv(local_filename)

    # Initialize a dictionary to act as the main data carrier container
    result = {}
    # Keep an untouched, unscaled copy of the full DataFrame for final backtesting calculations
    result["df"] = df.copy()

    # Verify that every single requested feature column actually exists in the working DataFrame
    for col in feature_columns:
        assert (
            col in df.columns
        ), f"'{col}' does not exist in the dataframe."  # Halt if a feature is missing

    # Convert the index dates into an explicit data column if it doesn't already exist
    if "date" not in df.columns:
        df["date"] = df.index

    # Requirement 1e: Scaler data structure and feature scaling execution
    if scale:
        # Create an internal dictionary to map feature names to their respective scaling models
        column_scaler = {}
        for column in feature_columns:
            # Instantiate an independent MinMaxScaler to force values into a strict (0, 1) boundary
            scaler = preprocessing.MinMaxScaler() 
            # Learn the Min and Max values ​​of that data column, perform a conversion calculation to the range (0, 1), 
            # and then assign the cleaned data back to overwrite the old column in the df table.
            df[column] = scaler.fit_transform( 
                np.expand_dims(df[column].values, axis=1) 
                 # Take the 1D array of values ​​from the current column
                 # and use the expand_dims function to insert a new axis at the column position (axis=1), 
                 # transforming the array from a one-dimensional array into a vertical 2D column array, 
                 # because the fit_transform function requires the input data to be a 2D array.
            )
            # Store the trained scaler instance linked to its column string name for future reference
            column_scaler[column] = scaler
        # Expose the dictionary of scalers out to the main result object
        result["column_scaler"] = column_scaler

    # Requirement 1b: Create the target column by shifting the price backwards
    df["future"] = df["adjclose"].shift(-lookup_step)

    # Temporarily isolate the final few feature records before they are removed by dropna
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # Requirement 1b: Drop any incomplete rows containing NaN cells generated by shifting
    df.dropna(inplace=True)

    # Initialize lists to process and build historical sequence blocks
    sequence_data = []
    # Create a double-ended queue with a locked maximum length to act as a moving window
    sequences = deque(maxlen=n_steps)

    # Iterate simultaneously over rows of combined feature lists and target labels
    for entry, target in zip(
        df[feature_columns + ["date"]].values, df["future"].values
    ):
        # Push the current row item into the right side of the sliding queue window
        sequences.append(entry)
        # Once the queue gathers enough consecutive history days, save the sequence block
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Construct the ultimate prediction seed array by concatenating the window tail with the lookahead gap
    last_sequence = list([s[: len(feature_columns)] for s in sequences]) + list(
        last_sequence
    )
    last_sequence = np.array(last_sequence).astype(np.float32)
    result["last_sequence"] = last_sequence

    # Separate out the prepared sequence datasets into explicit feature inputs (X) and labels (y)
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # Transform standard Python lists into raw, highly optimized NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Requirement 1c: Handle different data splitting strategies (Chronological vs Random)
    if split_by_date:
        # Calculate the hard boundary index dividing training records from testing records
        train_samples = int((1 - test_size) * len(X))
        # Assign historical older blocks to the training matrices
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        # Assign more recent blocks to the independent testing matrices
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        # If shuffling is enabled, randomize the internal sequence layout order within each set
        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # Perform an unconditional, non-linear random data split using scikit-learn
        (
            result["X_train"],
            result["X_test"],
            result["y_train"],
            result["y_test"],
        ) = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # Extract the original index timestamp references representing the testing blocks
    dates = result["X_test"][:, -1, -1]
    # Reconstruct a clean validation DataFrame tracking only test dates
    result["test_df"] = result["df"].loc[dates]
    # Filter out any redundant date indexes to keep validation measurements mathematically sound
    result["test_df"] = result["test_df"][
        ~result["test_df"].index.duplicated(keep="first")
    ]

    # Strip out trailing non-numeric date tags from the arrays and enforce standard float32 precision
    result["X_train"] = result["X_train"][:, :, : len(feature_columns)].astype(
        np.float32
    )
    result["X_test"] = result["X_test"][:, :, : len(feature_columns)].astype(
        np.float32
    )

    # Return the fully processed dictionary containing all datasets, dataframes, and scalers
    return result


# =================================================----------------------------
# TASK 3 FUNCTIONS (VISUALIZATION EXTENSIONS)
# =================================================----------------------------

def plot_candlestick(df, n_days=1):
    """
    Plots a professional stock market candlestick chart using the mplfinance library.
    Supports dynamic aggregation where each candle represents 'n' trading days.
    """
    plot_df = df.copy()
        
    if n_days > 1:
        # Group rows by integer division to cluster exactly 'n' sequential trading days together
        group_identifier = np.arange(len(plot_df)) // n_days
        
        # Apply strict financial aggregation rules across each n-day trading cluster
        plot_df = plot_df.groupby(group_identifier).agg({
            'open': 'first',   # The opening price of the first day in the cluster
            'high': 'max',     # The absolute maximum peak price reached during the n days
            'low': 'min',      # The absolute minimum bottom price dropped during the n days
            'close': 'last',   # The final closing price of the last day in the cluster
            'volume': 'sum'    # The cumulative total volume traded across all n days
        })
        
        # Map original timestamp references back onto the newly aggregated rows
        plot_df.index = df.index[::n_days][:len(plot_df)]

    plot_df.index = pd.to_datetime(plot_df.index, utc=True).tz_localize(None)

    # Configure custom visual cosmetics for the financial charting layout
    custom_market_style = mpf.make_mpf_style(base_mpf_style='charles', gridstyle='--')
    
    # Call the core plotting engine from the tutorial specifications
    mpf.plot(plot_df, 
             type='candle', 
             volume=True, 
             style=custom_market_style, 
             title=f"Candlestick Chart ({n_days}-Day Aggregation)")


def plot_moving_boxplot(df, n_days=30):
    """
    Displays stock market financial data using a boxplot chart across a specific
    moving window of 'n' consecutive trading days to capture historical volatility.
    """
    box_df = df.copy()
    
    # Slice out strictly the most recent 'n' consecutive rows to build the targeted moving window
    window_data = box_df.tail(n_days)
    
    # Isolate the core pricing columns that represent the target distributions
    target_columns = ['open', 'high', 'low', 'close']
    data_matrix = [window_data[col].values for col in target_columns]
    
    # Initialize a clean Matplotlib canvas structure for explicit render tracking
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Core plotting call to render the statistical distribution plots
    box_plots = ax.boxplot(data_matrix, patch_artist=True, notch=False, labels=target_columns)
    
    # Define aesthetic color palettes to distinguish individual price dimensions
    component_colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    for patch, color in zip(box_plots['boxes'], component_colors):
        patch.set_facecolor(color)  
        
    # Configure graph metadata labels to enhance readability for human end-users
    ax.set_title(f"Price Distribution Boxplot (Last {n_days} Consecutive Trading Days Window)")
    ax.set_ylabel("Price Asset Value ($)")
    ax.set_xlabel("Market Price Features")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()


# =================================================----------------------------
# TASK 4 FUNCTIONS (DYNAMIC MODEL FACTORY - INTEGRATED FROM P1)
# =================================================----------------------------

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False, output_steps=1):
    """
    Dynamically constructs a stacked recurrent neural network model based on provided hyperparameters.
    Supports flexible selection of cell types (LSTM, GRU, RNN), varying depth, and optional bidirectional wrapping.
    Updated for Task 5: supports multi-step output nodes via output_steps.
    """
    # Initialize a Sequential model container which allows us to linearize neural network layers one after another
    model = Sequential()
    
    # Execute a loop tracking from 0 up to 'n_layers - 1' to programmatically stack recurrent structures
    for i in range(n_layers):
        
        # Condition checking for the first structural layer of the deep neural network framework
        if i == 0:
            
            # Sub-branch handling when the user enables a wrapper to read sequences forward and backward simultaneously
            if bidirectional:
                # Add a Bidirectional layer factory that twins the hidden recurrent structures to capture both history and future contexts
                # 'batch_input_shape' enforces a strict 3D dimensional constraint (batch_size, time_steps, features) into Keras
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                # Define the rigid spatial entry point shape tracking historical lookback windows and available feature metrics
                model.add(Input(shape=(sequence_length, n_features)))
                # Instantiate the chosen recurrent cell type; 'return_sequences=True' is mandatory to pass a 3D hidden state block forward
                model.add(cell(units, return_sequences=True))
                
        # Condition checking for the final hidden recurrent layer positioned immediately prior to output projection
        elif i == n_layers - 1:
            
            # Sub-branch handling for the final layer when bidirectional sequence modeling is active
            if bidirectional:
                # Append a final bidirectional cell; 'return_sequences=False' squashes the time dimension to emit a 2D array
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                # Append a standard final recurrent cell; it flattens temporal sequences and outputs only the final internal step state
                model.add(cell(units, return_sequences=False))
                
        # Condition handling for all intermediate intermediate hidden layers positioned between the first and last tiers
        else:
            
            # Sub-branch handling for intermediate tiers when bidirectional sequence modeling is active
            if bidirectional:
                # Keep full sequence parameters intact across bidirectional channels by enforcing return_sequences=True
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                # Keep standard 3D sequential data vectors returning intact to serve as inputs for the next layer down the stack
                model.add(cell(units, return_sequences=True))
                
        # Inject an isolated regularization Dropout layer directly following each recurrent processing block
        # It randomly zero-out inputs at a fixed rate ('dropout') during training to suppress co-adaptation and prevent overfitting
        model.add(Dropout(dropout))
        
    # Append a final fully-connected Dense layer acting as a linear continuous mathematical regression projection node
    # Updated to output_steps for Task 5 multistep support.
    model.add(Dense(output_steps, activation="linear"))
    
    # Configure and anchor the structural properties of the network model prior to initializing weights
    # Binds the calculation tracker loss, historical assessment metrics, and gradient descent optimization algorithms
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    # Return the fully compiled, programmatically modularized Deep Learning architecture ready for fitting operations
    return model


# =================================================----------------------------
# TASK 5 FUNCTIONS (MULTIVARIATE & MULTISTEP DATA ENGINE)
# =================================================----------------------------

def load_multivariate_multistep_data(ticker, sequence_length=60, k_days=5, test_size=0.2,
                                    feature_columns=["adjclose", "volume", "open", "high", "low"],
                                    start_date="2020-01-01", end_date="2024-07-02",
                                    load_local=True, store_local=True):
    """
    Requirement 1, 2, and 3: Loads a dataset with multiple features (Multivariate)
    and creates target sequence blocks of 'k' days into the future (Multistep).
    """
    local_filename = f"data/{ticker}_{start_date}_to_{end_date}_multistep.csv"

    if load_local and os.path.exists(local_filename):
        print(f"[INFO] Loading multistep dataset locally from: {local_filename}")
        df = pd.read_csv(local_filename, index_col=0, parse_dates=True)
    else:
        print(f"[INFO] Fetching multistep data live from Yahoo Finance for: {ticker}")
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        df.columns = df.columns.str.lower()
        df["adjclose"] = df["close"]
        if store_local:
            if not os.path.isdir("data"): os.mkdir("data")
            df.to_csv(local_filename)

    result = {}
    result["df"] = df.copy()

    # Feature column verification
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # Normalize multiple columns independently (Multivariate requirement)
    column_scaler = {}
    for column in feature_columns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler
    result["column_scaler"] = column_scaler

    # Create multistep labels matrix: [t+1, t+2, ..., t+k]
    target_cols = []
    for i in range(1, k_days + 1):
        col_name = f"future_{i}"
        df[col_name] = df["adjclose"].shift(-i)
        target_cols.append(col_name)

    # Drop trailing NaN lines safely
    df.dropna(subset=target_cols, inplace=True)

    # Build sequence vectors
    X, y = [], []
    feature_matrix = df[feature_columns].values
    target_matrix = df[target_cols].values

    for i in range(sequence_length, len(df)):
        X.append(feature_matrix[i - sequence_length:i])
        y.append(target_matrix[i - 1])

    X, y = np.array(X), np.array(y)

    # Sequential chronological split to avoid forward lookup data leakage
    train_samples = int((1 - test_size) * len(X))
    result["X_train"], result["X_test"] = X[:train_samples].astype(np.float32), X[train_samples:].astype(np.float32)
    result["y_train"], result["y_test"] = y[:train_samples].astype(np.float32), y[train_samples:].astype(np.float32)

    return result


# =================================================----------------------------
# TASK 6 FUNCTIONS (ENSEMBLE MODELING APPROACH)
# =================================================----------------------------

def fit_predict_arima(train_series, test_series, order=(5, 1, 0)):
    """
    Requirement 1: Fits an ARIMA model and performs walk-forward validation 
    using REAL test market data at each time step.
    """
    history = list(train_series)
    arima_forecasts = []
    
    # Walk-forward testing: Input the actual value of the test set after each prediction step.
    for t in range(len(test_series)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        arima_forecasts.append(yhat)
        
        # Record the actual price of day t into the history to serve as a reference point for day t+1.
        history.append(test_series[t]) 
        
    return np.array(arima_forecasts)


def run_ensemble_pipeline(ticker, start_date="2020-01-01", end_date="2024-07-02", 
                          arima_order=(5, 1, 0), dl_cell=GRU, weight_dl=0.6):
    """
    Requirement 1 & 2: Combines a statistical analysis method (ARIMA) with the existing
    Deep Learning model (GRU/LSTM) using a dynamic weighted average ensemble approach.
    """
    # Reuse Task 5 data ingestion engine for multi-feature sequence mapping
    t5_data = load_multivariate_multistep_data(
        ticker=ticker, sequence_length=60, k_days=1, test_size=0.2,
        start_date=start_date, end_date=end_date
    )
    
    X_train, y_train = t5_data["X_train"], t5_data["y_train"]
    X_test, y_test = t5_data["X_test"], t5_data["y_test"]
    
    # 1. Execute Deep Learning Predictive Pipeline
    dl_model = create_model(
        sequence_length=60, n_features=X_train.shape[2], units=128,
        cell=dl_cell, n_layers=2, dropout=0.1, loss="huber", optimizer="adam", output_steps=1
    )
    print(f"[ENSEMBLE] Training Deep Learning Model component ({dl_cell.__name__})...")
    dl_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    dl_predictions = dl_model.predict(X_test).flatten()
    
    # 2. Execute Statistical ARIMA Predictive Pipeline
    close_scaler = t5_data["column_scaler"]["adjclose"]
    raw_train_prices = close_scaler.inverse_transform(y_train).flatten()
    raw_test_prices = close_scaler.inverse_transform(y_test).flatten() # Lấy chuỗi giá test thực tế
    
    print("[ENSEMBLE] Generating ARIMA statistical component rolling forecasts...")
    arima_raw_predictions = fit_predict_arima(raw_train_prices, raw_test_prices, order=arima_order)
    
    # Scale ARIMA outputs back to uniform coordinates for ensemble weight calculation
    arima_scaled_predictions = close_scaler.transform(arima_raw_predictions.reshape(-1, 1)).flatten()
    
    # 3. Apply Fusion Logic (Weighted Ensemble Formulation)
    weight_arima = 1.0 - weight_dl
    ensemble_scaled = (weight_dl * dl_predictions) + (weight_arima * arima_scaled_predictions)
    
    # Back-transform all individual output arrays to absolute real value metrics
    final_ensemble_prices = close_scaler.inverse_transform(ensemble_scaled.reshape(-1, 1)).flatten()
    final_dl_prices = close_scaler.inverse_transform(dl_predictions.reshape(-1, 1)).flatten()
    final_actual_prices = close_scaler.inverse_transform(y_test).flatten()
    
    # Compute system valuation summaries (MAE)
    mae_ensemble = np.mean(np.abs(final_ensemble_prices - final_actual_prices))
    mae_dl = np.mean(np.abs(final_dl_prices - final_actual_prices))
    mae_arima = np.mean(np.abs(arima_raw_predictions - final_actual_prices))
    
    print(f"\n[RESULTS] Baseline ARIMA MAE: {mae_arima:.4f}")
    print(f"[RESULTS] Baseline DL Model MAE: {mae_dl:.4f}")
    print(f"[RESULTS] Hybrid Ensemble Model MAE: {mae_ensemble:.4f}")
    
    return final_actual_prices, final_dl_prices, arima_raw_predictions, final_ensemble_prices


# =================================================----------------------------
# CORE PIPELINE EXECUTION BLOCK (TRADITIONAL v0.1 BASELINE FLOW)
# =================================================----------------------------

if __name__ == "__main__":
    #------------------------------------------------------------------------------
    # Load Data
    ## TO DO:
    # 1) Check if data has been saved before. 
    # If so, load the saved data
    # If not, save the data into a directory
    #------------------------------------------------------------------------------
    # DATA_SOURCE = "yahoo"
    COMPANY = 'CBA.AX'

    TRAIN_START = '2020-01-01'     # Start date to read
    TRAIN_END = '2023-08-01'       # End date to read

    # data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo

    # Get the data for the stock AAPL
    data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

    #------------------------------------------------------------------------------
    # Prepare Data
    ## To do:
    # 1) Check if data has been prepared before. 
    # If so, load the saved data
    # If not, save the data into a directory
    # 2) Use a different price value eg. mid-point of Open & Close
    # 3) Change the Prediction days
    #------------------------------------------------------------------------------
    PRICE_VALUE = "Close"

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    # Note that, by default, feature_range=(0, 1). Thus, if you want a different 
    # feature_range (min,max) then you'll need to specify it here
    scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1)) 
    # Flatten and normalise the data
    # First, we reshape a 1D array(n) to 2D array(n,1)
    # We have to do that because sklearn.preprocessing.fit_transform()
    # requires a 2D array
    # Here n == len(scaled_data)
    # Then, we scale the whole array to the range (0,1)
    # The parameter -1 allows (np.)reshape to figure out the array size n automatically 
    # values.reshape(-1, 1) 
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
    # When reshaping an array, the new shape must contain the same number of elements 
    # as the old shape, meaning the products of the two shapes' dimensions must be equal. 
    # When using a -1, the dimension corresponding to the -1 will be the product of 
    # the dimensions of the original array divided by the product of the dimensions 
    # given to reshape so as to maintain the same number of elements.

    # Number of days to look back to base the prediction
    PREDICTION_DAYS = 60 # Original

    # To store the training data
    x_train = []
    y_train = []

    scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
    # Prepare the data
    for x in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x-PREDICTION_DAYS:x])
        y_train.append(scaled_data[x])

    # Convert them into an array
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
    # and q = PREDICTION_DAYS; while y_train is a 1D array(p)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
    # is an array of p inputs with each input being a 2D array 

    #------------------------------------------------------------------------------
    # Build the Model
    ## TO DO:
    # 1) Check if data has been built before. 
    # If so, load the saved data
    # If not, save the data into a directory
    # 2) Change the model to increase accuracy?
    #------------------------------------------------------------------------------
    model = Sequential() # Basic neural network
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    # for some useful examples

    # Modern Keras 3 compatibility adjustment layer
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=100, return_sequences=True))
    # This is our first hidden layer which also spcifies an input layer. 
    # That's why we specify the input shape for this layer; 
    # i.e. the format of each training example
    # The above would be equivalent to the following two lines of code:
    # model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
    # model.add(LSTM(units=50, return_sequences=True))
    # For som eadvances explanation of return_sequences:
    # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
    # https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
    # As explained there, for a stacked LSTM, you must set return_sequences=True 
    # when stacking LSTM layers so that the next LSTM layer has a 
    # three-dimensional sequence input. 

    # Finally, units specifies the number of nodes in this layer.
    # This is one of the parameters you want to play with to see what number
    # of units will give you better prediction quality (for your problem)

    model.add(Dropout(0.1))
    # The Dropout layer randomly sets input units to 0 with a frequency of 
    # rate (= 0.1 above) at each step during training time, which helps 
    # prevent overfitting (one of the major problems of ML). 

    model.add(LSTM(units=50, return_sequences=True))
    # More on Stacked LSTM:
    # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))

    model.add(Dense(units=1)) 
    # Prediction of the next closing value of the stock price

    # We compile the model by specify the parameters for the model
    # See lecture Week 6 (COS30018)
    model.compile(optimizer='adam', loss='mean_squared_error')
    # The optimizer and loss are two important parameters when building an 
    # ANN model. Choosing a different optimizer/loss can affect the prediction
    # quality significantly. You should try other settings to learn; e.g.
        
    # optimizer='rmsprop'/'sgd'/'adadelta'/...
    # loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

    # Now we are going to train this model with our training data 
    # (x_train, y_train)
    model.fit(x_train, y_train, epochs=100, batch_size=64)
    # Other parameters to consider: How many rounds(epochs) are we going to 
    # train our model? Typically, the more the better, but be careful about
    # overfitting!
    # What about batch_size? Well, again, please refer to 
    # Lecture Week 6 (COS30018): If you update your model for each and every 
    # input sample, then there are potentially 2 issues: 1. If you training 
    # data is very big (billions of input samples) then it will take VERY long;
    # 2. Each and every input can immediately makes changes to your model
    # (a souce of overfitting). Thus, we do this in batches: We'll look at
    # the aggreated errors/losses from a batch of, say, 32 input samples
    # and update our model based on this aggregated loss.

    # TO DO:
    # Save the model and reload it
    # Sometimes, it takes a lot of effort to train your model (again, look at
    # a training data with billions of input samples). Thus, after spending so 
    # much computing power to train your model, you may want to save it so that
    # in the future, when you want to make the prediction, you only need to load
    # your pre-trained model and run it on the new input for which the prediction
    # need to be made.

    #------------------------------------------------------------------------------
    # Test the model accuracy on existing data
    #------------------------------------------------------------------------------
    # Load the test data
    TEST_START = '2023-08-02'
    TEST_END = '2024-07-02'

    # test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

    test_data = yf.download(COMPANY, TEST_START, TEST_END)

    # The above bug is the reason for the following line of code
    # test_data = test_data[1:]

    actual_prices = test_data[PRICE_VALUE].values

    total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    # We need to do the above because to predict the closing price of the fisrt
    # PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
    # data from the training period

    model_inputs = model_inputs.reshape(-1, 1)
    # TO DO: Explain the above line

    model_inputs = scaler.transform(model_inputs)
    # We again normalize our closing price data to fit them into the range (0,1)
    # using the same scaler used above 
    # However, there may be a problem: scaler was computed on the basis of
    # the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
    # but there may be a lower/higher price during the test period 
    # [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
    # greater than one)
    # We'll call this ISSUE #2

    # TO DO: Generally, there is a better way to process the data so that we 
    # can use part of it for training and the rest for testing. You need to 
    # implement such a way

    #------------------------------------------------------------------------------
    # Make predictions on test data
    #------------------------------------------------------------------------------
    x_test = []
    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # TO DO: Explain the above 5 lines

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    # Clearly, as we transform our data into the normalized range (0,1),
    # we now need to reverse this transformation 
    #------------------------------------------------------------------------------
    # Plot the test predictions
    ## To do:
    # 1) Candle stick charts
    # 2) Chart showing High & Lows of the day
    # 3) Show chart of next few days (predicted)
    #------------------------------------------------------------------------------

    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.show()

    #------------------------------------------------------------------------------
    # Predict next day
    #------------------------------------------------------------------------------

    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

    # A few concluding remarks here:
    # 1. The predictor is quite bad, especially if you look at the next day 
    # prediction, it missed the actual price by about 10%-13%
    # Can you find the reason?
    # 2. The code base at
    # https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
    # gives a much better prediction. Even though on the surface, it didn't seem 
    # to be a big difference (both use Stacked LSTM)
    # Again, can you explain it?
    # A more advanced and quite different technique use CNN to analyse the images
    # of the stock price changes to detect some patterns with the trend of
    # the stock price:
    # https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
    # Can you combine these different techniques for a better prediction??

    print("\n" + "="*50)
    print("[INTEGRATION TEST] Verifying newly integrated Task 2 & Task 3 functions...")
    print("="*50)
    
    # Executing Advanced Data Loader (Task 2)
    advanced_data = load_data(ticker=COMPANY, start_date=TRAIN_START, end_date=TRAIN_END)
    raw_dataframe = advanced_data['df']
    
    # Executing Visualization Modules (Task 3)
    print("-> Rendering Candlestick Chart (5-Day Aggregation)...")
    plot_candlestick(raw_dataframe, n_days=5)
    
    print("-> Rendering Moving Volatility Boxplot Chart (Last 45 Days)...")
    plot_moving_boxplot(raw_dataframe, n_days=45)

    # -------------------------------------------------------------------------
    # TASK 5 PIPELINE VERIFICATION (MULTIVARIATE & MULTISTEP FORECASTING)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("[TASK 5 EXECUTION] Training Multivariate & Multistep Model...")
    print("="*50)

    HORIZON_K = 5  # Predict 5 days into the future
    LOOKBACK_STEPS = 60
    FEATURE_COLS = ["adjclose", "volume", "open", "high", "low"]

    # Ingesting Task 5 sequence blocks
    t5_data = load_multivariate_multistep_data(
        ticker=COMPANY,
        sequence_length=LOOKBACK_STEPS,
        k_days=HORIZON_K,
        feature_columns=FEATURE_COLS,
        start_date="2020-01-01",
        end_date="2024-07-02"
    )

    X_train_m, y_train_m = t5_data["X_train"], t5_data["y_train"]
    X_test_m, y_test_m = t5_data["X_test"], t5_data["y_test"]

    # Instantiate dynamic model factory updated with output_steps dimension parameter
    t5_model = create_model(
        sequence_length=LOOKBACK_STEPS,
        n_features=len(FEATURE_COLS),
        units=128,
        cell=GRU,          # Optimized GRU architecture
        n_layers=2,
        dropout=0.1,
        loss="huber",      # Huber loss selection
        optimizer="adam",
        output_steps=HORIZON_K  # Dynamic output shape projection
    )

    print(f"[INFO] Fitting Task 5 model (Input features: {X_train_m.shape[2]} | Target Steps: {HORIZON_K})...")
    t5_model.fit(X_train_m, y_train_m, epochs=15, batch_size=32, validation_data=(X_test_m, y_test_m))

    # Evaluate multi-step forecasts
    predictions_m = t5_model.predict(X_test_m)

    # Invert price scaling vectors specifically for the target 'adjclose' dimension
    close_scaler = t5_data["column_scaler"]["adjclose"]
    real_prices_m = close_scaler.inverse_transform(y_test_m)
    forecasted_prices_m = close_scaler.inverse_transform(predictions_m)

    print(f"-> Task 5 tracking complete. Test predictions shape: {forecasted_prices_m.shape}")
    print(f"-> Sample Forecasted Prices for the next {HORIZON_K} days:\n{forecasted_prices_m[-1]}")

    # -------------------------------------------------------------------------
    # TASK 6 PIPELINE VERIFICATION (HYBRID ENSEMBLE SYSTEM)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("[TASK 6 EXECUTION] Running ARIMA + GRU Ensemble Pipeline...")
    print("="*50)
    
    # Execute structural ensemble evaluation using cross-method parameters
    act, dl_p, arima_p, ens_p = run_ensemble_pipeline(
        ticker=COMPANY, start_date="2020-01-01", end_date="2024-07-02",
        arima_order=(5, 1, 0), dl_cell=GRU, weight_dl=0.5
    )
    
    # Render final comparative visualization for validation
    plt.figure(figsize=(10, 5))
    plt.plot(act, color="black", label="Actual Price")
    plt.plot(dl_p, color="blue", alpha=0.6, label="Standalone GRU Component")
    plt.plot(arima_p, color="orange", alpha=0.6, label="Standalone ARIMA Component")
    plt.plot(ens_p, color="red", linewidth=2, label="Hybrid Ensemble Model")
    plt.title(f"{COMPANY} Share Price - Ensemble System Forecasting (v0.6)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()
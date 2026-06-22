import csv
import os
from datetime import datetime

import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint

from parameters import (
    BIDIRECTIONAL,
    END_DATE,
    FEATURE_COLUMNS,
    LOAD_LOCAL_DATA,
    LOCAL_DATA_PATH,
    LOOKUP_STEP,
    LOSS,
    NAN_METHOD,
    OPTIMIZER,
    SAVE_LOCAL_DATA,
    SCALE,
    SHUFFLE,
    START_DATE,
    TASK4_EXPERIMENTS,
    TEST_SIZE,
    ticker,
    ticker_clean,
    ticker_data_filename,
    date_now,
    N_STEPS,
    SPLIT_BY_DATE,
)
from stock_prediction import create_model, get_final_df, load_data, predict


def _ensure_directories():
    for folder in ["results", "logs", "data", "csv-results"]:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def _build_model_name(experiment):
    units_label = "x".join(str(unit) for unit in experiment["units"])
    base_name = (
        f"{date_now}_{ticker_clean}-{experiment['name']}-{LOSS}-{OPTIMIZER}-"
        f"{experiment['cell']}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{experiment['n_layers']}-"
        f"units-{units_label}"
    )
    return (
        base_name + "-b"
        if BIDIRECTIONAL
        else base_name
    )


def _summarize_experiment(model, data, history, model_name, experiment):
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    final_df = get_final_df(model, data)
    future_price = predict(model, data)
    accuracy_score = (
        len(final_df[final_df["sell_profit"] > 0]) + len(final_df[final_df["buy_profit"] > 0])
    ) / len(final_df)

    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    total_buy_profit = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    total_profit = total_buy_profit + total_sell_profit
    profit_per_trade = total_profit / len(final_df)

    csv_path = os.path.join("csv-results", model_name + ".csv")
    final_df.to_csv(csv_path)

    return {
        "experiment": experiment["name"],
        "cell": experiment["cell"],
        "n_layers": experiment["n_layers"],
        "units": str(experiment["units"]),
        "dropout": experiment["dropout"],
        "epochs": experiment["epochs"],
        "batch_size": experiment["batch_size"],
        "loss": float(loss),
        "mae": float(mae),
        "mean_absolute_error": float(mean_absolute_error),
        "accuracy_score": float(accuracy_score),
        "total_buy_profit": float(total_buy_profit),
        "total_sell_profit": float(total_sell_profit),
        "total_profit": float(total_profit),
        "profit_per_trade": float(profit_per_trade),
        "future_price": float(future_price),
        "best_val_loss": float(min(history.history.get("val_loss", [loss]))),
        "model_name": model_name,
    }


def main():
    _ensure_directories()

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

    data["df"].to_csv(ticker_data_filename)
    summaries = []

    for experiment in TASK4_EXPERIMENTS:
        model_name = _build_model_name(experiment)
        print(f"\n=== Running experiment: {experiment['name']} ===")

        model = create_model(
            N_STEPS,
            len(data["feature_columns"]),
            loss=LOSS,
            units=experiment["units"],
            cell=experiment["cell"],
            n_layers=experiment["n_layers"],
            dropout=experiment["dropout"],
            optimizer=OPTIMIZER,
            bidirectional=BIDIRECTIONAL,
        )

        checkpointer = ModelCheckpoint(
            os.path.join("results", model_name + ".weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        )
        history = model.fit(
            data["X_train"],
            data["y_train"],
            batch_size=experiment["batch_size"],
            epochs=experiment["epochs"],
            validation_data=(data["X_test"], data["y_test"]),
            callbacks=[checkpointer],
            verbose=1,
        )

        model.load_weights(os.path.join("results", model_name + ".weights.h5"))
        summary = _summarize_experiment(model, data, history, model_name, experiment)
        summaries.append(summary)
        print(summary)

    results_df = pd.DataFrame(summaries)
    results_path = os.path.join("csv-results", f"task4_experiment_summary_{date_now}.csv")
    results_df.to_csv(results_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nSaved summary to {results_path}")


if __name__ == "__main__":
    main()
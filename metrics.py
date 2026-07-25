"""
==========================================
TwinEV v2
Evaluation Metrics
==========================================
"""

import math
import numpy as np
from plot_style import set_style

set_style()

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)


def rmse(actual, predicted):

    return math.sqrt(
        mean_squared_error(actual, predicted)
    )


def mae(actual, predicted):

    return mean_absolute_error(
        actual,
        predicted
    )


def mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    return np.mean(

        np.abs(

            (actual - predicted)

            /

            (actual + 1e-8)

        )

    ) * 100


def control_metrics(power, hard_limit=150, soft_limit=130):

    power = np.array(power)

    return {

        "Energy Delivered (kWh)": round(power.sum(), 2),

        "Peak (kW)": round(power.max(), 2),

        "Mean Power (kW)": round(power.mean(), 2),

        "Variance": round(power.var(), 2),

        "Hard Overloads": int(np.sum(power > hard_limit)),

        "Soft Overloads": int(np.sum(power > soft_limit))

    }


def forecast_metrics(actual, predicted):

    return {

        "RMSE": round(rmse(actual, predicted), 2),

        "MAE": round(mae(actual, predicted), 2),

        "MAPE": round(mape(actual, predicted), 2)

    }


def compare_models(model_names, actual_list, prediction_list):

    results = []

    for name, pred in zip(model_names, prediction_list):

        m = forecast_metrics(actual_list, pred)

        m["Model"] = name

        results.append(m)

    return results
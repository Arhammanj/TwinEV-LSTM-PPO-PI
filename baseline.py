"""
==========================================
TwinEV v2
Baseline Controllers
==========================================
"""

import numpy as np


class FCFSBaseline:
    """
    First Come First Serve
    """

    def predict(self, demand):
        return np.array(demand)


class RoundRobinBaseline:
    """
    Simple Round Robin Controller
    """

    def __init__(self, max_power=150):
        self.max_power = max_power

    def predict(self, demand):

        demand = np.array(demand)

        if len(demand) == 0:
            return demand

        avg = np.mean(demand)

        power = np.full(len(demand), avg)

        power = np.clip(power, 0, self.max_power)

        return power


class GreedyBaseline:
    """
    Greedy Peak Limiting
    """

    def __init__(self, limit=130):
        self.limit = limit

    def predict(self, demand):

        demand = np.array(demand)

        return np.minimum(demand, self.limit)


def evaluate_baseline(name, actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    energy = predicted.sum()

    peak = predicted.max()

    overloads = np.sum(predicted > 150)

    soft_overloads = np.sum(predicted > 130)

    variance = np.var(predicted)

    return {

        "Model": name,

        "Energy Delivered (kWh)": round(energy, 2),

        "Peak (kW)": round(float(peak), 2),

        "Hard Overloads": int(overloads),

        "Soft Overloads": int(soft_overloads),

        "Variance": round(float(variance), 2)

    }
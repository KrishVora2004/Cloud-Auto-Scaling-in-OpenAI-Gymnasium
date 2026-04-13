import numpy as np


def compute_metrics(data):
    cost = np.sum([d.get("cost", 0) for d in data])

    error_rates = [d.get("error_rate", 0) for d in data]
    sla_violation = np.mean(error_rates)

    latency = np.mean([d.get("response_time", 0) for d in data])
    utilization = np.mean([d.get("utilization", 0) for d in data])

    return {
        "cost": cost,
        "sla_violation": sla_violation,
        "latency": latency,
        "utilization": utilization
    }


def aggregate(results):
    final = {}

    for key in results[0]:
        values = [r[key] for r in results]

        final[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    return final
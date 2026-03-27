# tests/test_workload.py

import matplotlib.pyplot as plt
from sim.cloud_sim import CloudSimulator


def test_workload_pattern():

    sim = CloudSimulator()

    lambdas = []

    for _ in range(500):
        metrics = sim.step(action=1)  # maintain
        lambdas.append(metrics["lambda"])

    plt.plot(lambdas)
    plt.title("Simulated Workload Pattern")
    plt.xlabel("Time Steps")
    plt.ylabel("Lambda (Request Rate)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_workload_pattern()



# to Run: python -m tests.test_workload
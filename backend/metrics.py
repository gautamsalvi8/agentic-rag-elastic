import numpy as np

class MetricsTracker:
    def __init__(self):
        self.latencies = []

    def add(self, latency):
        self.latencies.append(latency)

    def report(self):
        return {
            "avg": round(np.mean(self.latencies), 3),
            "p95": round(np.percentile(self.latencies, 95), 3)
        }

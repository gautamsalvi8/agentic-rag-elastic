import time
import statistics


class Timer:
    def __init__(self):
        self.times = {}
        self.starts = {}

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        self.times[name] = time.time() - self.starts[name]

    def report(self):
        print("\n⚡ Latency Report")
        for k, v in self.times.items():
            print(f"{k}: {v:.3f} sec")


# =========================================
# NEW → Multiple query benchmark
# =========================================

def run_query_benchmark(search, generator, queries):
    """
    Runs many queries and reports average latency
    """

    latencies = []

    print("\n🚀 Running Multi-Query Benchmark...\n")

    for q in queries:
        start = time.time()

        docs = search.search(q)
        _ = generator.generate(q, docs)

        latencies.append(time.time() - start)

    print("\n📊 Benchmark Results")
    print("--------------------")
    print("Queries:", len(queries))
    print("Avg latency:", round(statistics.mean(latencies), 3), "sec")
    print("Min latency:", round(min(latencies), 3), "sec")
    print("Max latency:", round(max(latencies), 3), "sec")
    print("P95 latency:", round(statistics.quantiles(latencies, n=20)[-1], 3), "sec")

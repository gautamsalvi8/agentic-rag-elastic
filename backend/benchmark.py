import time


class Timer:
    def __init__(self):
        self.times = {}

    def start(self, name):
        self.times[name] = time.time()

    def stop(self, name):
        self.times[name] = time.time() - self.times[name]

    def report(self):
        print("\n⚡ Latency Report")
        for k, v in self.times.items():
            print(f"{k}: {round(v, 3)} sec")

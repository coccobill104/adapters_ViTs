import csv, os, time

def append_result(path, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)

class Timer:
    def __init__(self):
        self.t0 = None
    def start(self):
        self.t0 = time.time()
    def stop(self):
        return time.time() - self.t0

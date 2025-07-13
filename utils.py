import time
import torch


class Timer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self):
        torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t
        self.mem = peak_memory()

def peak_memory():
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.max_memory_allocated() / 1024**2
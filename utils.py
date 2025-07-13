import time
import yaml
import torch

class Tracker:
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



def load_config(path=None, **kwargs):
    with open(path) as f:
        configs = yaml.safe_load(f)


    for k in ["batches", "in_dim", "out_dim", "hiddens", "depths"]:
        if k in kwargs and kwargs[k] is not None:
            configs[k] = kwargs[k]

    for k in ["batches", "hiddens", "depths"]:
        if isinstance(configs[k], str):
            configs[k] = [int(v) for v in configs[k].split(",")]

    return configs
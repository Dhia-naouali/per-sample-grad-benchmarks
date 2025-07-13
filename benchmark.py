import pandas as pd
import torch
from tqdm import tqdm
from .core import make_mlp, retain_graph_vmapped, vjp_vmapped
from .utils import Tracker

METHODS = {
    "retain_graph": retain_graph_vmapped,
    "vjp": vjp_vmapped,
}


def single_run(method, batch, hidden, depth, in_dim=8, out_dim=1):
    model = make_mlp(in_dim, hidden, depth, out_dim)
    x = torch.randn(batch, in_dim).cuda()
    y = torch.zeros(batch, out_dim).cuda()

    torch.cuda.empty_cache()
    with Tracker() as t:
        METHODS[method](model, x, y)

    return dict(
        method=method,
        batch=batch,
        hidden=hidden,
        depth=depth,
        time=t.dt,
        memory=t.mem,
    )


def sweep(
    batches=(16, 5),
    hiddens=(32, 4),
    depths=(2, 3),
):
    assert len(batches) == len(hiddens) == len(depths) == 2, "invalid sweep settings"

    def expand(setting):
        base, steps = setting
        return [base * 2**i for i in range(steps)]

    batches, hiddens, depths = expand(*batches), expand(*hiddens), expand(*depths)

    rows = []
    total = len(METHODS) * len(batches) * len(hiddens) * len(depths)
    with tqdm(total=total, desc="benchmark") as pbar:
        for method in METHODS:
            for b in batches:
                for h in hiddens:
                    for d in depths:
                        rows.append(single_run(method, b, h, d))
                        pbar.update(1)
                        
    return pd.DataFrame(rows)
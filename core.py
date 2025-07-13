import torch 
from torch import nn, func
import torch.nn.functional as F
    

def retain_graph_vmapped(model, x, y):
    model.zero_grad()
    logits = model(x)
    losses = F.mse_loss(logits, y, reduction="none").squeeze()
    identity = torch.eye(x.size(0), device=x.device)
    param_names, params = zip(*model.named_parameters())

    def compute_sample_grad(loss_selector):
        return torch.autograd.grad(
            losses, 
            params, 
            loss_selector,
            retain_graph=True
        )

    grads = func.vmap(compute_sample_grad)(identity)
    return [
        {name: grads[i][b] for i, name in enumerate(param_names)}
        for b in range(x.size(0))
    ]



def vjp_vmapped(model, x, y):
    model.zero_grad()
    params = dict(model.named_parameters())
    def forward(p, x):
        return func.functional_call(model, p, (x,))
    
    def criterion(p, x, y):
        return F.mse_loss(forward(p, x), y, reduction="none").squeeze()
    
    _, vjp_fn = func.vjp(criterion, params, x, y); del _
    identity = torch.eye(x.size(0), device=x.device)

    grads = func.vmap(lambda g: vjp_fn(g)[0])(identity)

    return [
        {name: grads[name][b] for name in grads}
        for b in range(x.size(0))
    ]


def init_mlp(in_dim, hidden_dim, out_dim, depth):
    layers = []
    for _ in range(depth-1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers).cuda()
import torch 
from torch import nn, func
import torch.nn.functional as F
    

def retain_graph_vmapped_autograd(model, x, y):
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



def vjp_vmapped_func(model, x, y):
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

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.Sigmoid(),
    nn.Linear(16, 32),
    nn.Sigmoid(),
    nn.Linear(32, 1)
).cuda()

x = torch.randn(12, 8).cuda()
y = torch.zeros(12, 1).cuda()

g1 = retain_graph_vmapped_autograd(model, x, y)
g2 = vjp_vmapped_func(model, x, y)
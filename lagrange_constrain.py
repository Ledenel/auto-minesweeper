from torch import nn
import torch


def flip(x):
    return -x


class LagrangeConstrainedLoss(nn.Module):
    def __init__(self, eq_zero: torch.Tensor = None, geq_zero: torch.Tensor = None):
        super().__init__()
        self.eq_zero = eq_zero
        self.geq_zero = geq_zero
        if eq_zero is not None:
            self.lagrange_multipliers_eq = nn.Parameter(torch.Tensor(eq_zero.shape))
            self.lagrange_multipliers_eq.register_hook(flip)
        if geq_zero is not None:
            self.lagrange_multipliers_geq = nn.Parameter(torch.Tensor(eq_zero.shape))
            self.lagrange_multipliers_geq.register_hook(flip)
            self.slack_vars = nn.Parameter(torch.Tensor(geq_zero.shape))

    def forward(self, loss):
        if self.eq_zero is not None:
            loss += self.lagrange_multipliers_eq * self.eq_zero
        if self.geq_zero is not None:
            slack = self.slack_vars * self.slack_vars
            ineq = self.geq_zero - slack
            loss += self.lagrange_multipliers_geq * ineq
        return loss

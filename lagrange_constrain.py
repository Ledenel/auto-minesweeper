from torch import nn
import torch


def flip(x):
    return -x


class LagrangeConstrainedLoss(nn.Module):
    def __init__(self, eqs_zero_len=0, geqs_zero_len=0):
        super().__init__()
        if eqs_zero_len > 0:
            self.lagrange_multipliers_eq = nn.Parameter(torch.Tensor(size=[eqs_zero_len]))
            self.lagrange_multipliers_eq.register_hook(flip)
            torch.nn.init.normal_(self.lagrange_multipliers_eq)
            # torch.nn.init.xavier_normal_(self.lagrange_multipliers_eq)
        if geqs_zero_len > 0:
            self.lagrange_multipliers_geq = nn.Parameter(torch.Tensor(size=[eqs_zero_len]))
            self.lagrange_multipliers_geq.register_hook(flip)
            self.slack_vars = nn.Parameter(torch.Tensor(size=[eqs_zero_len]))
            # torch.nn.init.xavier_normal_(self.slack_vars)
            # torch.nn.init.xavier_normal_(self.lagrange_multipliers_geq)

    def forward(self, loss, eq_zero: torch.Tensor = None, geq_zero: torch.Tensor = None):
        if eq_zero is not None:
            loss += (self.lagrange_multipliers_eq * eq_zero).sum()
        if geq_zero is not None:
            slack = self.slack_vars * self.slack_vars
            ineq = geq_zero - slack
            loss += (self.lagrange_multipliers_geq * ineq).sum()
        return loss

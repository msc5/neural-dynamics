from typing import Literal
import torch
import torch.nn as nn


class DynamicsModel (nn.Module):

    def __init__(self, size: int, device: Literal['cuda', 'cpu'] = 'cuda') -> None:
        super().__init__()
        self.device = device
        self.out_size = size
        self.A = nn.parameter.Parameter(torch.rand(size, size, device=device))

    def forward(self, t: torch.Tensor, initial: torch.Tensor):

        t = t.to(self.device)
        initial = initial.to(self.device)

        x = (self.A[None] * t[:, None, None]).exp()
        x = (x[None] @ initial[:, None, :, None]).squeeze()

        return x

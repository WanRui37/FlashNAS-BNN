import random

import torch


class PseudoWeights:
    def __init__(self, weights, ideal=None, nadir=None):
        super().__init__()
        self.weights = weights
        self.ideal = ideal
        self.nadir = nadir

    def _do(self, F):
        ideal, nadir = self.ideal, self.nadir

        if ideal is None:
            ideal, _ = torch.min(F, dim=0)
        if nadir is None:
            nadir, _ = torch.max(F, dim=0)

        pseudo_weights = (nadir.view(1, -1) - F) / (nadir.view(1, -1) - ideal.view(1, -1))
        # print(f"pseudo_weights.shape:{pseudo_weights.shape}")
        # print(f"torch.sum(pseudo_weights, dim=1).shape:{torch.sum(pseudo_weights, dim=1).shape}")

        pseudo_weights = pseudo_weights / torch.sum(pseudo_weights, dim=1).view(-1, 1)
        # print(f"pseudo_weights:{pseudo_weights}")

        I = torch.argmin(torch.sum(torch.abs(pseudo_weights - self.weights), dim=1))
        # print(f"pseudo_weights:{torch.sum(torch.abs(pseudo_weights - self.weights), dim=1)}")

        return I, pseudo_weights

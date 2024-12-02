import torch

from typing import List
from pointcept.models.dbbd.BaseClasses import AggregatorBase


class MaxPoolAggregator(AggregatorBase):
    def __init__(self):
        super(MaxPoolAggregator, self).__init__()

    def forward(self, features: torch.Tensor, dim=1) -> torch.Tensor:
        # features: (N, D)
        aggregated_feature, _ = torch.max(features, dim=dim)  # (B, D) if dim = 1 //  (D,) if dim = 0
        return aggregated_feature

    @property
    def method_name(self) -> str:
        return 'max_pool_aggregator'
from torch import nn


class EncoderBase(nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()

    def forward(self, x):
        raise NotImplementedError("EncoderBase is an abstract class.")


class AggregatorBase(nn.Module):
    def __init__(self):
        super(AggregatorBase, self).__init__()

    def forward(self, features):
        raise NotImplementedError("AggregatorBase is an abstract class.")


class FeaturePropagationBase(nn.Module):
    def __init__(self):
        super(FeaturePropagationBase, self).__init__()

    def propagate(self, parent_feature, current_feature):
        raise NotImplementedError("FeaturePropagationBase is an abstract class.")

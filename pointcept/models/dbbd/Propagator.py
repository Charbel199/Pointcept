import torch
from torch import nn
from pointcept.models.dbbd.BaseClasses import FeaturePropagationBase

# class ConcatPropagation(FeaturePropagationBase):
#     def __init__(self, feature_dim):
#         super(ConcatPropagation, self).__init__()
#         self.linear = nn.Linear(feature_dim * 2, feature_dim)
#         self.activation = nn.ReLU()

#     def propagate(self, parent_feature: torch.Tensor, current_feature: torch.Tensor) -> torch.Tensor:
#         if parent_feature is not None:
#             # Concatenate along the feature dimension
#             combined_feature = torch.cat([current_feature, parent_feature], dim=0)  # Shape: [2 * feature_dim]
#             # Pass through linear layer and activation
#             combined_feature = self.linear(combined_feature.unsqueeze(0))  # Shape: [1, feature_dim]
#             combined_feature = self.activation(combined_feature)
#             combined_feature = combined_feature.squeeze(0)  # Shape: [feature_dim]
#         else:
#             combined_feature = current_feature
#         return combined_feature
    

class ConcatPropagation(FeaturePropagationBase):
    def __init__(self):
        super(ConcatPropagation, self).__init__()

    def propagate(self, parent_feature: torch.Tensor, current_feature: torch.Tensor) -> torch.Tensor:
        if parent_feature is not None:
            # Concatenate along the feature dimension

            # CUSTOM LOGIC
            repeated_vector = parent_feature.unsqueeze(0).repeat(current_feature.size(0), 1)  # (levelN, C)
            # Concatenate along the second dimension
            combined_feature = torch.cat((current_feature, repeated_vector), dim=1)  # (levelN, C+D)

            combined_feature = self.linear_block(combined_feature)  # Shape: [levelN, feature_dim]
            combined_feature = self.activation(combined_feature)


            # # OLD LOGIC
            # combined_feature = torch.cat([current_feature, parent_feature], dim=0)  # Shape: [2 * feature_dim]
            # # Pass through linear layer and activation
            # combined_feature = self.linear(combined_feature.unsqueeze(0))  # Shape: [1, feature_dim]
            # combined_feature = self.activation(combined_feature)
            # combined_feature = combined_feature.squeeze(0)  # Shape: [feature_dim]
        else:
            combined_feature = current_feature
        return combined_feature

    def update_feature_dim(self, input_dim, feature_dim):
        self.feature_dim = feature_dim
        self.linear_block = nn.Linear(input_dim, feature_dim)
        self.activation = nn.ReLU()

    @property
    def method_name(self) -> str:
        return 'concat_propagation'
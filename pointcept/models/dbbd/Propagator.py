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
            # parent_feature: shape = [96] [output_dim]
            # current_feature: shape = [5000, 3] [min_num_points_per_pointcloud, D]
            # CUSTOM LOGIC
            # shape: [500, 96] [min_num_points_per_pointcloud, output_dim]
            repeated_vector = parent_feature.unsqueeze(0).repeat(current_feature.size(0), 1)
            original_dim = 3  # Original feature size (before padding)
            #shape: [500, 3]
            current_feature = current_feature[:, :original_dim]  # Slice only the original features

            # Concatenate along the second dimension
            # shape: [500, 99] [min_num_points_per_pointcloud, (output_dim + D)]
            combined_feature = torch.cat((current_feature, repeated_vector), dim=1)
            # shape: [500, 96] [min_num_points_per_pointcloud, D]
            combined_feature = self.linear(combined_feature)
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
        self.linear = nn.Linear(input_dim, feature_dim)
        self.activation = nn.ReLU()

    @property
    def method_name(self) -> str:
        return 'concat_propagation'
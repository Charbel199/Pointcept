import torch
from torch import nn
from pointcept.models.dbbd.BaseClasses import EncoderBase
from point_transformer_pytorch import PointTransformerLayer
from PointTransformerV3.model import PointTransformerV3


class Encoder(EncoderBase):
    def __init__(self, input_dim:int, fixed_feature_dim: int, output_dim: int):
        super(Encoder, self).__init__()
        self.fixed_feature_dim = fixed_feature_dim  # Fixed size after the first layer
        self.output_dim = output_dim  # Output feature dimension
        #self.first_layers = nn.ModuleDict()  # Store first layers keyed by input dimension
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, fixed_feature_dim),
            nn.ReLU(),  # Activation after first layer
            nn.Linear(fixed_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.shared_layers(x)
        return x



class PointTransformerEncoder(EncoderBase):
    def __init__(self, input_dim: int, fixed_feature_dim: int, output_dim: int):
        super(PointTransformerEncoder, self).__init__()
        self.fixed_feature_dim = fixed_feature_dim  # Fixed size after the first layer
        self.output_dim = output_dim  # Output feature dimension
        # self.first_layers = nn.ModuleDict()  # Store first layers keyed by input dimension

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, fixed_feature_dim),
            nn.ReLU(),
            nn.Linear(fixed_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

        self.attn = PointTransformerLayer(
            dim=output_dim,
            pos_mlp_hidden_dim=fixed_feature_dim,
            attn_mlp_hidden_mult=4,
            num_neighbors=None
        )

    def forward(self, x):
        mask = torch.ones(1, x.shape[0]).bool().to(x.device)
        x_pos = x[:, :3].unsqueeze(0)
        # x_feat = x[:,3:].unsqueeze(0)  # color

        y = self.shared_layers(x)

        x = self.attn(y.unsqueeze(0), x_pos, mask=mask)
        return x.squeeze(0)


class PointTransformerv3Encoder(EncoderBase):
    def __init__(self, input_dim:int, fixed_feature_dim: int, output_dim: int):
        super(PointTransformerv3Encoder, self).__init__()
        # self.fixed_feature_dim = fixed_feature_dim  # Fixed size after the first layer
        # self.output_dim = output_dim  # Output feature dimension
        # #self.first_layers = nn.ModuleDict()  # Store first layers keyed by input dimension

        # self.shared_layers = nn.Sequential(
        #     nn.Linear(input_dim, fixed_feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(fixed_feature_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, output_dim),
        #     nn.ReLU()
        # )

        self.transformer = PointTransformerV3(in_channels=input_dim) #, pre_norm=False, pdnorm_decouple=False, pdnorm_affine=False)
        self.linear = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        shape of x: [N, F] where N is the number of points, F the number of features (3 or 6)
        """
        N, F = x.shape
        data = {"feat": x[:, 3:], "coord": x[:, :3], "grid_size": 0.5, "offset": torch.tensor([N//2, N]).to(x.device)}
        x = self.transformer(data)["feat"]
        x = self.linear(x)
        return x


class PointTransformerv3EncoderBatch(EncoderBase):
    def __init__(self, input_dim: int, fixed_feature_dim: int, output_dim: int):
        super(PointTransformerv3EncoderBatch, self).__init__()
        # self.fixed_feature_dim = fixed_feature_dim  # Fixed size after the first layer
        # self.output_dim = output_dim  # Output feature dimension
        # #self.first_layers = nn.ModuleDict()  # Store first layers keyed by input dimension

        # self.shared_layers = nn.Sequential(
        #     nn.Linear(input_dim, fixed_feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(fixed_feature_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, output_dim),
        #     nn.ReLU()
        # )

        self.transformer = PointTransformerV3(
            in_channels=input_dim)  # , pre_norm=False, pdnorm_decouple=False, pdnorm_affine=False)
        self.linear = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        shape of x: List of Tensors<[N, F]> where N is the number of points, F the number of features (3 or 6)
        """
        shapes = torch.zeros(len(x), dtype=torch.int32)
        for i, pc in enumerate(x):
            shapes[i] = pc.shape[0]

        offset = torch.cumsum(shapes, dim=0).to(x.device)
        point_all = torch.cat(x, dim=0)  # produces tensor of shape (N1+N2+...+NM, F)

        data = {"feat": point_all[:, 3:], "coord": point_all[:, :3], "grid_size": 0.5, "offset": offset}
        x = self.transformer(data)["feat"]
        x = self.linear(x)

        # Split tensor again
        x = torch.split(x, shapes.tolist())
        return x
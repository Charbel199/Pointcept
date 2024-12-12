_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 2  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True

mx_lvl = 0
num_samples_per_level=2
point_max=20000
min_num_points_list = [0]
# model settings
model = dict(
    type="DBBD-v1m1",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=20,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    output_dim=20,
    device = "cuda",
    loss_method = "point",
    num_samples_per_level=num_samples_per_level,
    max_levels=mx_lvl,
)

# scheduler settings
epoch = 600
optimizer = dict(type="SGD", lr=0.1, momentum=0.8, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split=["train", "val", "test"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(
                type="ContrastiveViewsGenerator",
                view_keys=("coord", "color", "normal", "origin_coord"),
                view_trans_cfg=[
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=1),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=1),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="RandomColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.02,
                        p=0.8,
                    ),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=("origin_coord", "coord", "color", "normal"),
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                ],
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "view1_origin_coord",
                    "view1_grid_coord",
                    "view1_coord",
                    "view1_color",
                    "view1_normal",
                    "view2_origin_coord",
                    "view2_grid_coord",
                    "view2_coord",
                    "view2_color",
                    "view2_normal",
                ),
                offset_keys_dict=dict(
                    view1_offset="view1_coord", view2_offset="view2_coord"
                ),
                view1_feat_keys=("view1_color", "view1_normal"),
                view2_feat_keys=("view2_color", "view2_normal"),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

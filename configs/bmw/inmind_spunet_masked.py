_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 32  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = False
num_worker=1
evaluate = False
mx_lvl = 1
num_samples_per_level=2
point_max=500
min_num_points_list = [250, 125]
# model settings
model = dict(
    type="DBBD-v1m1",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    output_dim=13,
    device = "cuda",
    loss_method = "point_and_level",
    num_samples_per_level=num_samples_per_level,
    max_levels=mx_lvl,
    
    # NOTE Masked Variables added NOTE #
    backbone_in_channels=6,
    backbone_out_channels=96,
    mask_grid_size=0.1,
    mask_rate=0.4,
    view1_mix_prob=0.4,
    view2_mix_prob=0.4,
    matching_max_k=8,
    matching_max_radius=0.03,
    matching_max_pair=4096,
    nce_t=0.07,
    contrast_weight=1,
    reconstruct_weight=1,
    reconstruct_color=False,
    reconstruct_normal=False,
)

# scheduler settings
epoch = 3000
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")

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
        # num_samples_per_level=num_samples_per_level,
        # max_levels=mx_lvl,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="SphereCrop", point_max=point_max),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(
                type="ContrastiveViewsGenerator",
                view_keys=("coord", "color", "normal", "origin_coord"),
                view_trans_cfg=[
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], always_apply=True),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                                dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnvs",
                mode="train",
                keys=("origin_coord", "coord", "color", "normal"),
                return_grid_coord=True,
            ),
                    
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor")
                ],
            ),
            dict(type="DBDD", num_samples_per_level=num_samples_per_level, max_levels=mx_lvl, min_num_points_list=min_num_points_list),

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
                    "regions",
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
    # val=dict(
    #     type=dataset_type,
    #     split="Area_5",
    #     num_samples_per_level=num_samples_per_level,
    #     max_levels=mx_lvl,
    #     data_root=data_root,
    #     transform=[
    #         dict(type="CenterShift", apply_z=True),
    #         dict(type="SphereCrop", point_max=point_max),
    #         dict(
    #             type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
    #         ),
    #         dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    #         dict(
    #             type="ContrastiveViewsGenerator",
    #             view_keys=("coord", "color", "normal", "origin_coord"),
    #             view_trans_cfg=[
    #                 dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], always_apply=True),
    #                 dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    #                 dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    #                 dict(type="RandomScale", scale=[0.9, 1.1]),
    #                 dict(type="RandomFlip", p=0.5),
    #                 dict(type="RandomJitter", sigma=0.005, clip=0.02),
    #                 dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    #                 dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    #                 dict(type="ChromaticJitter", p=0.95, std=0.05),
    #                 dict(type="Copy", keys_dict={"coord": "grid_coord"}),
    #                 dict(type="CenterShift", apply_z=False),
    #                 dict(type="NormalizeColor")
    #             ],
    #         ),
    #         dict(type="DBDD",num_samples_per_level=num_samples_per_level,max_levels=mx_lvl,min_num_points_list=min_num_points_list),
    #         dict(type="ToTensor"),
    #         dict(
    #             type="Collect",
    #             keys=(
    #                 "view1_origin_coord",
    #                 "view1_grid_coord",
    #                 "view1_coord",
    #                 "view1_color",
    #                 "view1_normal",
    #                 "view2_origin_coord",
    #                 "view2_grid_coord",
    #                 "view2_coord",
    #                 "view2_color",
    #                 "view2_normal",
    #                 "regions",
    #             ),
    #             feat_keys=("color", "normal"),
    #         ),
    #     ],
    #     test_mode=False,
    # ),
    # test=dict(
    #     type=dataset_type,
    #     split="Area_5",
    #     data_root=data_root,
    #     num_samples_per_level=num_samples_per_level,
    #     max_levels=mx_lvl,
    #     transform=[
    #         dict(type="CenterShift", apply_z=True),
    #         dict(type="SphereCrop", point_max=point_max),
    #         dict(type="NormalizeColor"),
    #     ],
    #     test_mode=True,
    #     test_cfg=dict(
    #         voxelize=dict(
    #             type="GridSample",
    #             grid_size=0.02,
    #             hash_type="fnv",
    #             mode="test",
    #             keys=("coord", "color", "normal"),
    #             return_grid_coord=True,
    #         ),
    #         crop=None,
    #         post_transform=[
    #             dict(type="CenterShift", apply_z=False),
    #             dict(type="ToTensor"),
    #             dict(
    #                 type="Collect",
    #                 keys=("coord", "grid_coord", "index", "color"),
    #                 feat_keys=("normal", "normal"),
    #             ),
    #         ],
    #         aug_transform=[
    #             [dict(type="RandomScale", scale=[0.9, 0.9])],
    #             [dict(type="RandomScale", scale=[0.95, 0.95])],
    #             [dict(type="RandomScale", scale=[1, 1])],
    #             [dict(type="RandomScale", scale=[1.05, 1.05])],
    #             [dict(type="RandomScale", scale=[1.1, 1.1])],
    #             [
    #                 dict(type="RandomScale", scale=[0.9, 0.9]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #             [
    #                 dict(type="RandomScale", scale=[0.95, 0.95]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #             [
    #                 dict(type="RandomScale", scale=[1, 1]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #             [
    #                 dict(type="RandomScale", scale=[1.05, 1.05]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #             [
    #                 dict(type="RandomScale", scale=[1.1, 1.1]),
    #                 dict(type="RandomFlip", p=1),
    #             ],
    #         ],
    #     ),
    # ),
)


hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

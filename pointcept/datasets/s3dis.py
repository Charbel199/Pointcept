"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
from .dataset_with_hierarchical_regions import HierarchicalRegionsDataset
import os
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class S3DISDataset(DefaultDataset):
# class S3DISDataset(HierarchicalRegionsDataset):
    def get_data_name(self, idx):
        remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(remain)
        return f"{area_name}-{room_name}"

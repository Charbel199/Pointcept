import os
import pickle
import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS
from pointcept.utils.cache import shared_dict
from .preprocessing.hierarchical_region_proposal import hierarchical_region_proposal


@DATASETS.register_module()
class HierarchicalRegionsDataset(DefaultDataset):

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        print(f"Working on data_path: {data_path}")
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            print(f"Returning cache {cache_name}")
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                if not asset.endswith(".npz"):
                    continue
                else:
                    data_dict["regions"] = np.load(os.path.join(data_path, asset), allow_pickle=True)["region_dict"]
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)
            print(f"data_dict['coord'] : {data_dict['coord'].shape}")
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        ###########################
        #  Make hierarchical regions of the dataset. Store them to either npy or pickle file to be loaded later.
        ###########################
        num_samples_per_level = self.num_samples_per_level 
        max_levels = self.max_levels
        regions_path = os.path.join(data_path, f"regions_{num_samples_per_level}_{max_levels}.pickle")
        
        # # Load hierarchical_regions (pickle version)
        # if os.path.exists(regions_path):
        #     with open(regions_path, 'rb') as f:
        #         data_dict["regions"] = pickle.load(f)
        # else:
        regions = hierarchical_region_proposal(data_dict["coord"],data_dict["color"], 
            num_samples_per_level=num_samples_per_level, max_levels=max_levels, batch_idx=0)
        data_dict["regions"] = regions
            # with open(regions_path, 'wb') as f:
            #     pickle.dump(regions, f)

        return data_dict

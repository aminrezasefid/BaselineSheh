import os
import os.path as osp
import sys

from typing import Callable, List, Optional, Dict

import torch
from torch import Tensor
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)
from torch_geometric.transforms import Compose
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm
from torch_geometric.nn.models.schnet import qm9_target_dict


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


URLS = {
    "precise3d": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
}


class QM9(InMemoryDataset):
    def __init__(self, 
                 root: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False,
                 structure: str = "precise3d",
                 dataset_args: List[str] = None):
        self.structure = structure
        self.raw_url = URLS[structure]
        self.labels = dataset_args if dataset_args is not None else list(qm9_target_dict.keys())

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())
import torch
from torch_geometric.transforms import Compose
from .qm8_geometric import QM8_geometric
from typing import Dict

qm8_target_dict: Dict[int, str] = {
    0:"E1-CC2",
    1:"E2-CC2",
    2:"f1-CC2",
    3:"f2-CC2",
    4:"E1-PBE0",
    5:"E2-PBE0",
    6:"f1-PBE0",
    7:"f2-PBE0",
    12:"E1-CAM",
    13:"E2-CAM",
    14:"f1-CAM",
    15:"f2-CAM"
}

class QM8(QM8_geometric):
    def __init__(self, root, transform=None, dataset_arg=None, structure = None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm8_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(qm8_target_dict.values(), qm8_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM8, self).__init__(root, transform=transform, structure=structure)

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM8, self).download()

    def process(self):
        super(QM8, self).process()

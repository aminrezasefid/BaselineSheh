import torch
from torch_geometric.transforms import Compose
from .qm7_geometric import QM7_geometric
from typing import Dict

qm7_target_dict: Dict[int, str] = {
    0: 'u0_atom'
}

class QM7(QM7_geometric):
    def __init__(self, root, transform=None, dataset_arg=None, structure = None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm7_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(qm7_target_dict.values(), qm7_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM7, self).__init__(root, transform=transform, structure=structure)

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
        print("Downloading QM7 dataset...")
        super(QM7, self).download()

    def process(self):
        super(QM7, self).process()

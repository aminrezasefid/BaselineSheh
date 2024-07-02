from typing import Optional, Callable

import torch
from torch_geometric.transforms import Compose

import torchmdnet.datasets.embedded_molecule_net as emn

esol_target_dict = {
    0: "ESOL predicted log solubility in mols per litre",
    1: "Minimum Degree",
    2: "Molecular Weight",
    3: "Number of H-Bond Donors",
    4: "Number of Rings",
    5: "Number of Rotatable Bonds",
    6: "Polar Surface Area",
    7: "measured log solubility in mols per litre",
}


class ESOL:
    def __new__(cls, root, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None, dataset_arg=None,
                structure: emn.EmbeddingType = emn.EmbeddingType.TWO_D.value):
        if structure == emn.EmbeddingType.TWO_D.value:
            return _ESOL2D(root, transform, pre_transform, pre_filter, dataset_arg)
        elif structure == emn.EmbeddingType.THREE_D.value:
            return _ESOL3D(root, transform, pre_transform, pre_filter, dataset_arg)
        elif structure == emn.EmbeddingType.THREE_D_OPTIMIZED.value:
            return _ESOL3DOptimized(root, transform, pre_transform, pre_filter, dataset_arg)
        else:
            raise ValueError(f"Unavailable structure {structure} for ESOL dataset.")


class _ESOL(emn.EmbeddedMoleculeNet):
    """
    Do not call this class directly. Instead, create an instance of ESOL with the desired structure.
    """

    NAME = 'esol'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(esol_target_dict.values())}.'
        )

        # print the name of this class (its child's name)

        self.label = dataset_arg
        label2idx = dict(zip(esol_target_dict.values(), esol_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])
        super().__init__(root, transform, pre_transform, pre_filter)
        self.label = dataset_arg

    def get_name_column(self) -> str:
        return 'Compound ID'

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


class _ESOL2D(_ESOL, emn.EmbeddedMoleculeNet2DMixin):
    """
    Do not call this class directly. Instead, create an instance of ESOL with the desired structure.
    """
    POS_DATASET_URL = "https://drive.usercontent.google.com/download?id=1-AJkkZQ50oFcjLj1RGRlYpoMKPBLT2YY&export" \
                      "=download&authuser=2&confirm=t&uuid=8fcd7a44-e08c-4b85-88b3-f689adcadc7d&at" \
                      "=APZUnTWLKEGkfDb5QWhOFGQJaafd:1719918560896"

    def download(self):
        super().download()

    def process(self):
        super().process()


class _ESOL3D(_ESOL, emn.EmbeddedMoleculeNet3DMixin):
    """
    Do not call this class directly. Instead, create an instance of ESOL with the desired structure.
    """
    POS_DATASET_URL = "https://drive.usercontent.google.com/download?id=1SFh4vhLqXyCBEF2tdTiYzWXeXr8IkbHN&export" \
                      "=download&authuser=2&confirm=t&uuid=81806a61-5ee9-473a-bdc6-d708fca371b7&at" \
                      "=APZUnTUuAdduE92ZwgFLFhDE3QLL:1719918542495"

    def download(self):
        super().download()

    def process(self):
        super().process()


class _ESOL3DOptimized(_ESOL, emn.EmbeddedMoleculeNet3DOptimizedMixin):
    """
    Do not call this class directly. Instead, create an instance of ESOL with the desired structure.
    """
    POS_DATASET_URL = "https://drive.usercontent.google.com/download?id=1SFh4vhLqXyCBEF2tdTiYzWXeXr8IkbHN&export" \
                      "=download&authuser=2&confirm=t&uuid=81806a61-5ee9-473a-bdc6-d708fca371b7&at" \
                      "=APZUnTUuAdduE92ZwgFLFhDE3QLL:1719918542495"

    def download(self):
        super().download()

    def process(self):
        super().process()

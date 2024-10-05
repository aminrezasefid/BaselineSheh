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

qm8_target_dict: Dict[int, str] = {
    0: "E1-CC2",
    1: "E2-CC2",
    2: "f1-CC2",
    3: "f2-CC2",
    4: "E1-PBE0",
    5: "E2-PBE0",
    6: "f1-PBE0",
    7: "f2-PBE0",
    8: "E1-CAM",
    9: "E2-CAM",
    10: "f1-CAM",
    11: "f2-CAM",
}

URLS = {
    "precise3d": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz",
    "rdkit3d": "https://drive.google.com/uc?export=download&id=1hc0c--8cMbDlfplmJwRUXsfAq-Pj4daY",
    "optimized3d": "https://drive.google.com/uc?export=download&id=1qTxsIJ_aP7Qow6On4BfpBk9SorIzEh9a",
    "rdkit2d": "https://drive.google.com/uc?export=download&id=1MAqVW5GyHmLKLJqdDrIMIBxjDPW3ce3o",
    "pubchem": "https://drive.google.com/file/d/1NHcYEOcl6ZN4wUmBM60lB9YOZa_tZ6FG/view?usp=sharing",
}


class QM8(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        structure: str = "precise3d",
        dataset_args: List[str] = None,
    ):
        self.structure = structure
        self.raw_url = URLS[structure]
        self.labels = (
            dataset_args if dataset_args is not None else list(qm8_target_dict.keys())
        )

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa

            return (
                ["gdb8.sdf", "gdb8.sdf.csv"]
                if self.structure != "precise3d"
                else ["qm8.sdf", "qm8.sdf.csv"]
            )
        except ImportError:
            return ImportError("Please install 'rdkit' to download the dataset.")

    @property
    def processed_file_names(self) -> str:
        return "data_v3.pt"

    # @property
    # def target_column(self) -> List[str]:
    #     return [qm8_target_dict[label] for label in self.labels]

    def download(self):
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_tar(file_path, self.raw_dir)
            os.unlink(file_path)

        except ImportError:
            print("Please install 'rdkit' to download the dataset.", file=sys.stderr)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT

            RDLogger.DisableLog("rdApp.*")

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(
                (
                    "Using a pre-processed version of the dataset. Please "
                    "install 'rdkit' to alternatively process the raw data."
                ),
                file=sys.stderr,
            )

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "S": 5}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], "r") as f:
            if self.structure == "precise3d":
                target = [
                    [float(x) for x in line.split(",")[1:]]
                    for line in f.read().split("\n")[1:-1]
                ]
                target = [x[:8] + x[12:] for x in target]
            else:
                target = [
                    [float(x) for x in line.split(",")]
                    for line in f.read().split("\n")[1:-1]
                ]
            y = torch.tensor(target, dtype=torch.float)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            # check if any two atoms are overlapping
            # if torch.unique(pos, dim=0).size(0) != N:
            #     print(f"Skipping molecule {mol.GetProp('_Name')} as it contains overlapping atoms.")
            #     continue

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float
                )
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            name = mol.GetProp("_Name")

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def _filter_label(self, batch):
        if self.labels:
            batch.y = batch.y[:, self.labels]
        return batch

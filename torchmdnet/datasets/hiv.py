from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# # -*- coding: utf-8 -*-
# """temp.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/18elFaucW6izV1-JIpKUk1iCuKrt-vRkP
# """

# !pip install torch_geometric
# !pip install rdkit
# !pip install python-magic
# !pip install gdown

import sys
import torch
from torch import Tensor

#from torch_geometric.utils import one_hot, scatter

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}
############## Initiate Class #################################
class HIV_geometric(InMemoryDataset):

    raw_url = ('https://drive.google.com/uc?export=download&id=17gLTR88EOj_d0vUgaOM00dyR-oLBjSb4') #Dataset

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        #force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform=None, pre_filter=None)
                         #force_reload=force_reload)
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['HIV.sdf', 'HIV.sdf.csv'] #Dataset3D, CSV and unwanted
        except ImportError:
            return ['HIV.pt'] #Pre-processed

    @property
    def processed_file_names(self) -> str:
        return 'HIV_processed.pt'

    def download(self) -> None:
        import rdkit  # noqa
        import gdown  # you need to install this library
        #import magic
        print("Starting download...")
        file_path = download_url(self.raw_url, self.raw_dir)
        # Use gdown to download the file
        gdown.download(self.raw_url, output=file_path, quiet=False)

        print(f"Downloaded file to {file_path}")

        # # Check the file format
        # file_format = magic.from_file(file_path)
        # print(f"File format: {file_format}")

        # if 'zip' in file_format.lower():
        extract_zip(file_path, self.raw_dir)
        print(f"Extracted files to {self.raw_dir}")
        os.unlink(file_path)
        print("Deleted the zip file")
        # else:
        #     print("The file is not a zip file.")

        print("Download successful!")


    def process(self) -> None:
        try: # If RDKit installed
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')  # type: ignore
            WITH_RDKIT = True

        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT: # in RDKit not installed
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0]) # The pre-processed data is loaded from self.raw_paths[0] (raw_file_names[0])
            data_list = [Data(**data_dict) for data_dict in data_list] # The loaded data is a list of dictionaries, each representing a data sample. These dictionaries are converted to Data objects using a list comprehension

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0]) # The processed data list is saved to self.processed_paths[0] (processed_file_names[0])
            return
        # This part of the process function is responsible for processing the raw molecular data
        # and converting it into a format suitable for graph-based machine learning models
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S':5, 'Cl':6, 'P':7, 'Br':8} # The types dictionary maps atom types (H, C, N, O, F) to integers, and the
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3} # bonds dictionary maps bond types (single, double, triple, aromatic) to integers

        # The target values for each molecule (properties to be predicted) are loaded from a file (self.raw_names[1] = CSV).
        # These values are converted to a PyTorch tensor (y), rearranged, and scaled by a conversion factor.
        with open(self.raw_paths[1], 'r') as f:
            print("Reading csv", self.raw_paths[1])
            lines = f.read().split('\n')
            target = []
            for line in lines[1:]:  # This will skip the header row
                if line:  # This will skip the blank lines
                    target.append([int(line.split(',')[2])])  # HIV has 2 attributes

            y = torch.tensor(target, dtype=torch.float)
            print("targets set!")


            # y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            # y = y * conversion.view(1, -1)
        # Some molecules may be excluded from the dataset for various reasons (e.g., they are too large or have unusual properties).
        # The indices of these molecules are loaded from a file (self.raw_paths[2]) and stored in the skip list.
        # with open(self.raw_paths[2], 'r') as f: #Unwanteds -> skip list
        #     skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        # Create a molecule supplier: The Chem.SDMolSupplier object (suppl) is created to read molecules from a file (self.raw_paths[0])
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, # (raw_file_names[0] = SDF)
                                   sanitize=False)
        print("suppl done, start FOR MOL")
        # Each molecule in the supplier is processed in turn. If a molecule’s index is in the skip list, it is skipped
        data_list = []

        for i, mol in enumerate(tqdm(suppl)): # For all mols, skip unwanted and get info on others
            # if i in skip:
            #     continue
            # The number of atoms (N) and their 3D positions (pos) are obtained
            # print(f"{i}_th mol in progress")
            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []

            # For each atom, its type, atomic number, aromaticity, hybridization state, and number of hydrogen neighbors are recorded
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)
            # For each bond, its start and end atoms and bond type are recorded
            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            # The atom and bond information is used to construct the node features (x), edge indices (edge_index),
            # and edge attributes (edge_attr) for the graph representation of the molecule
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            # The molecule’s name and SMILES string are also recorded
            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            # All this information, for each molecule, is packaged into a Data object and added to the data_list
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

###### subclass HIV #########################
from torch_geometric.transforms import Compose
from typing import Dict
# import HIV as HIV_geometric

HIV_target_dict: Dict[int, str] = {
    0: 'activity',
    1: 'HIV_active'}

class HIV(HIV_geometric):
    def __init__(self, root, transform=None, dataset_arg=None, conf_num=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(HIV_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(HIV_target_dict.values(), HIV_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(HIV, self).__init__(root, transform=transform)

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
        super(HIV, self).download()

    def process(self):
        super(HIV, self).process()

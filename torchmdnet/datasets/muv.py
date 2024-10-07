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
    extract_zip,
)
from torch_geometric.transforms import Compose
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm
import errno
def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


import gdown
URLS = {
    "precise3d": "https://drive.google.com/uc?export=download&id=1W7EFQlgCyK9nLjcl275ATjv_pJZqBwOe",
    "optimized3d": "https://drive.google.com/uc?export=download&id=157WRUy7Ul-QZVQfiqDMSZXVtfjIYBlhn", ###### CHANGE ######
    "rdkit3d": "https://drive.google.com/uc?export=download&id=1JUCEK3OYCOfkZvhABgM2pkeebfVCB8jX",
    "rdkit2d": "https://drive.google.com/uc?export=download&id=1PY-3pCqlEbycJMYgeEaDQllMzQlOa97E"
}

muv_target_dict = {'MUV-466': 0, 'MUV-548': 1, 'MUV-600': 2, 'MUV-644': 3, 'MUV-652': 4, 'MUV-689': 5, 'MUV-692': 6, 'MUV-712': 7, 'MUV-713': 8, 'MUV-733': 9, 'MUV-737': 10, 'MUV-810': 11, 'MUV-832': 12, 'MUV-846': 13, 'MUV-852': 14, 'MUV-858': 15, 'MUV-859': 16}
def gdown_download_url(id: str, folder: str, log: bool = True):
    filename = id+".zip"
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {id}', file=sys.stderr)

    makedirs(folder)

    data = gdown.download(id=id,output=filename)


    return path
class MUV(InMemoryDataset): ###### CHANGE ######
    def __init__(self, 
                 root: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False,
                 structure: str = "rdkit3d",
                 dataset_args: List[str] = None):
        self.structure = structure
        self.raw_url = URLS[structure]
        self.labels = [muv_target_dict[label] for label in dataset_args] if dataset_args is not None else list(muv_target_dict.values()) ###### CHANGE ######

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
    
    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            file_names = {
                "precise3d": ['pubchem.sdf', 'pubchem.sdf.csv'],
                "optimized3d": ['rdkit_opt.sdf', 'rdkit_opt.sdf.csv'],
                "rdkit3d": ['rdkit_3D.sdf', 'rdkit_3D.sdf.csv'],          ###### CHANGE ######
                "rdkit2d": ['rdkit_graph.sdf', 'rdkit_graph.sdf.csv']
            }
            return file_names[self.structure]
        except ImportError:
            return ImportError("Please install 'rdkit' to download the dataset.")

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'
    
    def download(self):
        try:
            import rdkit  # noqa
            #import gdown
            file_path = gdown_download_url(self.raw_url, self.raw_dir)
            #gdown.download(self.raw_url, output=file_path, quiet=False)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

        except ImportError:
            print("Please install 'rdkit' to download the dataset.", file=sys.stderr)

    def process(self) -> None:
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return
 
        types = {'C': 0, 'N': 1, 'O': 2, 'H': 3, 'S': 4, 'Cl': 5, 'F': 6, 'Br': 7}   ###### CHANGE ######
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, BT.DATIVE: 4}


        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x) if x != '-100' and x != '' else -1
                       for x in line.split(',')]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            ## Create a mask for the diagonal
            # mask = torch.eye(N, dtype=bool)
            
            # # Compute the pairwise distances
            # distances = torch.cdist(pos, pos)
            
            # # Apply the mask to the distances (this will set the diagonal elements to infinity)
            # distances.masked_fill_(mask, float('inf'))
            
            # # Now check for overlapping atoms
            # if not torch.all(distances > 0):
            #     #print(f"Skipping molecule {i} due to overlapping atoms.")
            #     continue
              
            # check if any two atoms are overlapping
            # if torch.unique(pos, dim=0).size(0) != N:
            #     # print(f"Skipping molecule {mol.GetProp('_Name')} as it contains overlapping atoms.")
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
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            name = mol.GetProp('_Name')
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

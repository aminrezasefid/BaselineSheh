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
class BACE(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,types=None,bonds=None,
                 pre_transform: Optional[Callable] = None,num_confs=1,
                 pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None):
        self.types=types
        self.bonds=bonds
        self.num_confs=num_confs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def processed_file_names(self) -> str:
        return ['data_v3.pt','broken_smiles.pt']
    @property
    def raw_file_names(self) -> List[str]:
        return ['bace.csv']
    @property
    def target_column(self) -> str:
        return ['Class']
    def get_MMFF_mol(self,mol,numConfs=1):
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        except:
            return None
        return new_mol
    def process(self):
        df=pd.read_csv(self.raw_paths[0])
        self.smiles_list=list(df["mol"])
        target = df[self.target_column]

        target = target.replace(0, -1)  # convert 0 to -1
        target = target.fillna(0)  

        target = torch.tensor(target.values,dtype=torch.float)
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

            torch.save(self.collate(data_list), self.processed_paths[0])
            return
        data_list = []
        broken_smiles=[]
        non_conf_count=0
        for i, smile in enumerate(tqdm(self.smiles_list)):
            mol = AllChem.MolFromSmiles(smile)
            mol = self.get_MMFF_mol(mol,self.num_confs)
            if mol is None:
                non_conf_count+=1
                broken_smiles.append(smile)
                continue
            N = mol.GetNumAtoms()
            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
            z = torch.tensor(atomic_number, dtype=torch.long)
            name=smile
            confNums=mol.GetNumConformers()
            for confId in range(confNums):
                pos=mol.GetConformer(confId).GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)
                upos_num=np.unique(pos,axis=0).shape[0]
                pos_num=pos.shape[0]
                if upos_num!=pos_num:
                    non_conf_count+=1
                    broken_smiles.append(smile)
                    continue
                data = Data(z=z, pos=pos,y=target[i].unsqueeze(0), name=f"{confId}-{name}", idx=i)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
        print(f"{str(non_conf_count)} smiles couldn't generate conformer.")

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(broken_smiles,self.processed_paths[1])
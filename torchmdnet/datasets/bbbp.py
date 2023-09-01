from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
from glob import glob
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
class TOXCAST(InMemoryDataset):
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
        return ['bbbp.csv']
    @property
    def target_column(self) -> str:
        return ['p_np']
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
        self.smiles_list=list(df["smiles"])
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
        if self.types is None:
            types = {'O':0,'N':1,'C':2,'Cl':3,'Si':4,'Br':5,'Ba':6,'Nd':7,
            'Dy':8,'In':9,'P':10,'Sb':11,'Co':12,'S':13,'K':14,'Na':15,
            'B':16,'Ca':17,'Hg':18,'Ni':19,'Se':20,'Tl':21,'Cd':22,'F':23,
            'Fe':24,'Li':25,'Yb':26,'I':27,'Cr':28,'Sn':29,'Zn':30,'Cu':31,
            'Pb':32,'As':33,'Bi':34,'H':35,'Gd':36,'V':37,'Mn':38,'Au':39,'Ti':40,
            'Zr':41,'Mo':42,'Mg':43,'Eu':44,'Al':45,'Pt':46,'Sr':47,'Sc':48,
            'Ag':49,'Pd':50,'Be':51,'Ge':52,}
        if self.bonds is None:
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        data_list = []
        broken_smiles=[]
        non_conf_count=0
        idx=0
        for i, smile in enumerate(tqdm(self.smiles_list)):
            mol = AllChem.MolFromSmiles(smile)
            mol = self.get_MMFF_mol(mol,self.num_confs)
            if mol is None:
                non_conf_count+=1
                broken_smiles.append(smile)
                continue
            N = mol.GetNumAtoms()
            
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
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            #y = target[i].unsqueeze(0)
            #name = mol.GetProp('_Name')
            name=smile
            #data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
            #            edge_attr=edge_attr, y=y, name=name, idx=i)
            confNums=mol.GetNumConformers()
            for confId in range(confNums):
                pos=mol.GetConformer(confId).GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)
                data = Data(x=x, z=z, pos=pos, edge_index=edge_index,y=target[i].unsqueeze(0),
                            edge_attr=edge_attr, name=f"{confId}-{name}", idx=idx)
                idx+=1
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
        print(f"{str(non_conf_count)} smiles couldn't generate conformer.")

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(broken_smiles,self.processed_paths[1])
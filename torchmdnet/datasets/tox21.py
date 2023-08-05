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
class TOX21(InMemoryDataset):
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
        return ['tox21.csv']
    @property
    def target_column(self) -> str:
        return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
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
            types = {'C':0,'O':1,'N':2,'S':3,'P':4,'Cl':5,'I':6,
            'Zn':7,'F':8,'Ca':9,'As':10,'Br':11,'B':12,'H':13,'K':14,
            'Si':15,'Cu':16,'Mg':17,'Hg':18,'Cr':19,'Zr':20,
            'Sn':21,'Na':22,'Ba':23,'Au':24,'Pd':25,'Tl':26,'Fe':27,
            'Al':28,'Gd':29,'Ag':30,'Mo':31,'V':32,'Nd':33,'Co':34,'Yb':35,
            'Pb':36,'Sb':37,'In':38,'Li':39,'Ni':40,'Bi':41,'Cd':42,'Ti':43,
            'Se':44,'Dy':45,'Mn':46,'Sr':47,'Be':48,'Pt':49,'Ge':50,}
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
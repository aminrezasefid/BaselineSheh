from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
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
def get_MMFF_mol(mol,numConfs=1):
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        except:
            return None
        return new_mol
def worker(smiles_list,target,num_confs,procnum, return_dict,pre_filter,pre_transform):
        data_list=[]
        broken_smiles=[]
        for i, smile in enumerate(tqdm(smiles_list,position=procnum)):
            mol = AllChem.MolFromSmiles(smile)
            mol = get_MMFF_mol(mol,num_confs)
            if mol is None:
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
                    broken_smiles.append(smile)
                    continue
                data = Data(z=z, pos=pos,y=target[i].unsqueeze(0), name=f"{confId}-{name}", idx=i)

                if pre_filter is not None and not pre_filter(data):
                    continue
                if pre_transform is not None:
                    data = pre_transform(data)

                data_list.append(data)
        #print(f"{str(non_conf_count)} smiles couldn't generate conformer.")
        return_dict[str(procnum)+"data"]=data_list
        return_dict[str(procnum)+"broken"]=broken_smiles
        # torch.save(self.collate(data_list), self.processed_paths[0])
        # torch.save(broken_smiles,self.processed_paths[1])
class PCBA(InMemoryDataset):
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
        return ['pcba.csv']
    @property
    def target_column(self) -> str:
        return [
    'PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457',
    'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469',
    'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688',
    'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242',
    'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546',
    'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676', 'PCBA-411',
    'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294', 'PCBA-485297',
    'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349', 'PCBA-485353',
    'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208',
    'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339', 'PCBA-504444',
    'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842', 'PCBA-504845',
    'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317', 'PCBA-588342',
    'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591',
    'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233', 'PCBA-602310',
    'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171', 'PCBA-624173',
    'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288', 'PCBA-624291',
    'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644',
    'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104', 'PCBA-652105',
    'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979', 'PCBA-720504',
    'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553', 'PCBA-720579',
    'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709', 'PCBA-720711',
    'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881', 'PCBA-883',
    'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899', 'PCBA-902',
    'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915', 'PCBA-924',
    'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995'
    ]


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
        data_list = []
        broken_smiles=[]
        non_conf_count=0
        cpu_nums=mp.cpu_count()
        each_share=len(self.smiles_list)//cpu_nums
        last_share=each_share+len(self.smiles_list)-cpu_nums*each_share
        procs=[]
        from multiprocessing import Process
        manager = mp.Manager()
        shared_dict = manager.dict()
        data_list=[]
        for i in range(cpu_nums-1):
            l_bound=i*each_share
            u_bound=l_bound+each_share
            proc=Process(target=worker,args=(self.smiles_list[l_bound:u_bound],
                                            target[l_bound:u_bound],
                                            self.num_confs,
                                            i,
                                            shared_dict,
                                            self.pre_transform,
                                            self.pre_filter,))
            procs.append(proc)
            proc.start()
        l_bound=(cpu_nums-1)*each_share
        u_bound=l_bound+last_share
        proc=Process(target=worker,args=(self.smiles_list[l_bound:u_bound],
                                            target[l_bound:u_bound],
                                            self.num_confs,
                                            cpu_nums-1,
                                            shared_dict,
                                            self.pre_transform,
                                            self.pre_filter,))
        procs.append(proc)
        proc.start()
        for proc in procs:
            proc.join()
        for i in range(cpu_nums):
            data_list.extend(shared_dict[str(i)+"data"])
            broken_smiles.extend(shared_dict[str(i)+"broken"])
        print(f"{str(len(broken_smiles))} smiles couldn't generate conformer.")
        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(broken_smiles,self.processed_paths[1]) 

        
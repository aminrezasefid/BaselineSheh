import os
from copy import deepcopy
from enum import Enum
from typing import Optional, Callable, List

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdDistGeom import EmbedMolecule, ETKDGv3
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from torch_geometric.data import (Data, download_url)
from torch_geometric.datasets import MoleculeNet as MoleculeNet_geometric
from torch_geometric.datasets.molecule_net import x_map, e_map
from torch_scatter import scatter
from tqdm import tqdm


class EmbeddingType(Enum):
    PRECISE_THREE_D = 'Precise3D'
    TWO_D = '2D'
    THREE_D = '3D'
    THREE_D_OPTIMIZED = '3DOptimized'

    @staticmethod
    def all():
        # return the .value for all the enum values, using inline lambda function
        return list(map(lambda c: c.value, EmbeddingType))


class EmbeddedMoleculeNet(MoleculeNet_geometric):
    NAME: str = None
    POS_DATASET_URL: str = None

    def __init__(self, root, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        assert self.NAME is not None, "NAME must be set in the subclass"
        super().__init__(root, self.NAME, transform, pre_transform, pre_filter)

    def create_required_directories(self) -> None:
        self.create_required_directories()
        os.makedirs(os.path.join(self.processed_dir, 'created_structures/'), exist_ok=True)

    @staticmethod
    def get_name_column() -> str:
        pass

    @staticmethod
    def get_smiles_column() -> str:
        return 'smiles'

    def add_extra_properties(self, data: Data, dataset_row):
        """
        override this function to add extra properties to the extracted data during dataset processing
        :param data: the extracted Data object
        :param dataset_row: the currently-getting-processed row in the dataset
        :return:  appends the related data and returns the final data object
        """
        return

    def process(self):
        from rdkit import Chem
        from rdkit.Chem.rdchem import HybridizationType
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit import RDLogger

        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'P': 7, 'I': 8, 'Br': 9}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # assert csv file format extension
        assert self.raw_paths[0].endswith('.sdf'), "The raw file on index 0 must be a sdf file"
        assert self.raw_paths[1].endswith('.csv'), "The raw file on index 1 must be a csv file"

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        with open(self.raw_paths[1], 'r') as f:
            target = pd.read_csv(f).iloc[:, 1:].values
            target = target[:, [str(x).replace('.', '', 1).replace('-', '', 1).isdigit() for x in target[0]]]
            target = [[float(x) for x in line] for line in target]
            target = torch.tensor(target, dtype=torch.float)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            N = mol.GetNumAtoms()

            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
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
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')

            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def get_all_molecules(self):
        """
        :return: returns a list of Mol objects
        """
        result = []
        for row in self:
            # retrieve processed data
            smiles = row.smiles
            name = row.name

            # create molecule object
            mol = Chem.MolFromSmiles(smiles)
            mol.SetProp("_Name", smiles)
            mol.SetProp("_Compound_ID", name)  # unnecessary at the moment!

            mol = Chem.AddHs(mol)
            Compute2DCoords(mol)

            result.append(mol)
        return result

    @property
    def raw_file_names(self) -> List[str]:
        super_result = super().raw_file_names
        result = [self.get_pos_dataset_name()]
        if isinstance(super_result, str):
            result.append(super_result)
        else:
            result.extend(super_result)
        return result

    def download(self):
        super().download()
        if self.POS_DATASET_URL is not None:
            path = download_url(self.POS_DATASET_URL, self.raw_dir)
            # rename it to the desired name
            os.rename(path, os.path.join(self.raw_dir, self.get_pos_dataset_name()))

    def create_embedded_dataset(self, path):
        """
        each of the children shall implement this method, creating sdf files based on their embedding strategy
        :param path: the path to save the embedded sdf file
        :return:
        """
        pass

    def get_embedding_type(self) -> EmbeddingType:
        """
        to be implemented by all the children
        :return: returns a string containing the embedding type (e.g. 2D, 3D, 3D-Optimized)
        """
        pass

    @staticmethod
    def get_dataset_extension() -> str:
        return '.sdf'

    def get_pos_dataset_name(self) -> str:
        return self.NAME + '_' + self.get_embedding_type().value + self.get_dataset_extension()

    def get_embedded_dataset_path(self):
        dataset_name = self.get_pos_dataset_name()
        return os.path.join(self.raw_dir, dataset_name)

    @staticmethod
    def embed_molecule(molecule, random_seed: bool = False):
        mol = deepcopy(molecule)
        params = ETKDGv3()
        params.randomSeed = -1 if random_seed else 312
        convergence = EmbedMolecule(mol, params)
        while convergence < 0:
            print(f"failed to embed {mol.GetProp('_Name')}. retrying with random seeds...")
            params.randomSeed = -1
            params.maxAttempts = 5000
            convergence = EmbedMolecule(mol, params)
        return mol

    @staticmethod
    def optimize_embedding(input_mol, max_attempt: int = 500):
        molecule = deepcopy(input_mol)
        for attempt in range(max_attempt):
            Chem.SanitizeMol(molecule)
            convergence = 1
            max_iter = 200
            while convergence == 1:
                convergence = MMFFOptimizeMolecule(molecule, maxIters=max_iter)
                max_iter <<= 2
            if convergence == 0:
                if attempt != 0:
                    print(f"successfully optimized embedding for {molecule.GetProp('_Name')} after {attempt} retries!")
                return molecule
            if attempt < max_attempt - 1:
                molecule = EmbeddedMoleculeNet.embed_molecule(molecule, random_seed=True)
        print(f"optimization did not converge for molecule {molecule.GetProp('_Name')}. we will make do with the "
              f"un-optimized molecule")
        return input_mol

    def execute_embedded_dataset_creation(self):
        self.create_embedded_dataset(self.get_embedded_dataset_path())


class EmbeddedMoleculeNet2DMixin(EmbeddedMoleculeNet):
    def create_embedded_dataset(self, path):
        molecules = self.get_all_molecules()
        with Chem.SDWriter(path) as w:
            for _, mol in tqdm(enumerate(molecules), desc=f"Computing Type-4 2D structures ({self.NAME})"):
                m = deepcopy(mol)
                w.write(m)

    def get_embedding_type(self) -> EmbeddingType:
        return EmbeddingType.TWO_D


class EmbeddedMoleculeNet3DMixin(EmbeddedMoleculeNet):
    def create_embedded_dataset(self, path):
        molecules = self.get_all_molecules()
        with Chem.SDWriter(path) as w:
            for _, mol in tqdm(enumerate(molecules), desc=f"Computing Type-2 3D structures ({self.NAME})"):
                m = self.embed_molecule(mol)
                w.write(m)

    def get_embedding_type(self) -> EmbeddingType:
        return EmbeddingType.THREE_D


class EmbeddedMoleculeNet3DOptimizedMixin(EmbeddedMoleculeNet):
    def create_embedded_dataset(self, path):
        molecules = self.get_all_molecules()
        with Chem.SDWriter(path) as w:
            for _, mol in tqdm(enumerate(molecules), desc=f"Computing Type-3 3D structures ({self.NAME})"):
                m = self.embed_molecule(mol)
                m = self.optimize_embedding(m)
                w.write(m)

    def get_embedding_type(self) -> EmbeddingType:
        return EmbeddingType.THREE_D_OPTIMIZED


def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = [x_map['atomic_num'].index(atom.GetAtomicNum()),
                          x_map['chirality'].index(str(atom.GetChiralTag())),
                          x_map['degree'].index(atom.GetTotalDegree()),
                          x_map['formal_charge'].index(atom.GetFormalCharge()),
                          x_map['num_hs'].index(atom.GetTotalNumHs()), x_map['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()), x_map['hybridization'].index(str(atom.GetHybridization())),
                          x_map['is_aromatic'].index(atom.GetIsAromatic()), x_map['is_in_ring'].index(atom.IsInRing())]
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = [e_map['bond_type'].index(str(bond.GetBondType())), e_map['stereo'].index(str(bond.GetStereo())),
             e_map['is_conjugated'].index(bond.GetIsConjugated())]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

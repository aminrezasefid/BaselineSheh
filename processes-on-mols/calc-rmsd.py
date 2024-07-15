import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

QM7_SKIP_LIST = [
    '1 2.753415 1.686911 2.122795',
    '1 4.940981 0.903782 0.860442',
    '1 5.189535 2.297423 -0.368037',
    '1 1.964094 4.093345 0.737567',
]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rmsd(mol_pair):
    mol_precise, mol_other = mol_pair
    try:
        if mol_precise is None or mol_other is None:
            return None
        rdMolAlign.AlignMol(mol_other, mol_precise)
        return rdMolAlign.GetBestRMS(mol_other, mol_precise)
    except Exception as e:
        # logging.error(f"Error calculating RMSD for molecule {mol_precise.GetProp("_Name")} and {mol_other.GetProp("_Name")}: {e}")
        return None
    
def construct_mol_pairs(mols_precise, mols_other):
    mol_pairs = []
    for mol_precise in mols_precise:
        for mol_other in mols_other:
            if mol_precise.GetProp("_Name") == mol_other.GetProp("_Name"):
                mol_pairs.append((mol_precise, mol_other))
                break
    return mol_pairs

def process_molecules(dataset, precise_path, rdkit_3d_path, optimized_3d_path, structure_2d_path):
    # Read molecules from SDF files
    logging.info("Reading molecules from SDF files...")
    mols_precise = [mol for mol in Chem.SDMolSupplier(precise_path, removeHs=False,sanitize=False) if mol is not None]
    mols_rdkit_3d = [mol for mol in Chem.SDMolSupplier(rdkit_3d_path, removeHs=False,sanitize=False) if mol is not None]
    mols_optimized_3d = [mol for mol in Chem.SDMolSupplier(optimized_3d_path, removeHs=False,sanitize=False) if mol is not None]
    mols_2d = [mol for mol in Chem.SDMolSupplier(structure_2d_path, removeHs=False,sanitize=False) if mol is not None]

    # Working with QM7 requires a whole different approach so

    # if dataset == 'QM7':
    #     mols_precise_smiles = [Chem.MolToSmiles(mol, isomericSmiles = False) for mol in mols_precise]
    #     mols_2d_smiles = [Chem.MolToSmiles(mol, isomericSmiles = False) for mol in mols_2d]

    #     # Find common molecules by SMILES and get their indices
    #     logging.info("Finding common molecule SMILES and their indices...")
    #     common_indices = [(mols_precise_smiles.index(smiles), mols_2d_smiles.index(smiles)) for smiles in mols_precise_smiles if smiles in mols_2d_smiles]

    #     # Chech that the first index in the tuple is always greater equal than the second
    #     # return true if this is the case for all tuples
    #     if all([i >= j for i,j in common_indices]):
    #         logging.info("Indices are in the correct order")

    #     # Filter the precise and 2D structures based on the common indices
    #     logging.info("Filtering precise and 2D structures based on common indices...")
    #     mols_precise_filtered = [mols_precise[i] for i, _ in common_indices]
    #     mols_2d_filtered = [mols_2d[i] for _, i in common_indices]
    #     mols_3d_filtered = [mols_rdkit_3d[i] for _,i in common_indices]
    #     mols_3d_optimized_filtered = [mols_optimized_3d[i] for _,i in common_indices]

    # else:
    #     # Create a dictionary to match molecules by name
    #     logging.info("Creating dictionaries to match molecules by name...")
    #     mols_2d_dict = {mol.GetProp("_Name"): idx for idx, mol in enumerate(mols_2d)}

    #     # Find common molecules by name and get their indices
    #     logging.info("Finding common molecule names and their indices...")
    #     common_indices = [mols_2d_dict[mol.GetProp("_Name")] for mol in mols_rdkit_3d if mol.GetProp("_Name") in mols_2d_dict]
        
    #     # Filter the precise and 2D structures based on the common indices
    #     logging.info("Filtering precise and 2D structures based on common indices...")
    #     mols_precise_filtered = [mols_precise[i] for i in common_indices]
    #     mols_2d_filtered = [mols_2d[i] for i in common_indices]
    #     mols_3d_filtered = mols_rdkit_3d
    #     mols_3d_optimized_filtered = mols_optimized_3d

    # Prepare pairs of molecules for RMSD calculation
    logging.info("Preparing pairs of molecules for RMSD calculation...")
    # pairs_rdkit_3d = list(zip(mols_precise_filtered, mols_3d_filtered))
    # pairs_optimized_3d = list(zip(mols_precise_filtered, mols_3d_optimized_filtered))
    # pairs_2d = list(zip(mols_precise_filtered, mols_2d_filtered))

    pairs_rdkit_3d = construct_mol_pairs(mols_precise, mols_rdkit_3d)
    pairs_optimized_3d = construct_mol_pairs(mols_precise, mols_optimized_3d)
    pairs_2d = construct_mol_pairs(mols_precise, mols_2d)

    # Utilize multiprocessing for parallel processing
    logging.info("Starting parallel RMSD calculation...")
    with Pool(cpu_count()) as pool:
        rmsd_rdkit_3d = list(tqdm(pool.map(calculate_rmsd, pairs_rdkit_3d), total=len(pairs_rdkit_3d), desc='RMSD_rdkit_3d'))
        rmsd_optimized_3d = list(tqdm(pool.map(calculate_rmsd, pairs_optimized_3d), total=len(pairs_optimized_3d), desc='RMSD_optimized_3d'))
        rmsd_2d = list(tqdm(pool.map(calculate_rmsd, pairs_2d), total=len(pairs_2d), desc='RMSD_2d'))


    # Create DataFrame to store results
    logging.info("Storing results in a DataFrame...")
    rmsd_df = pd.DataFrame({
        'RMSD_rdkit_3d': rmsd_rdkit_3d,
        'RMSD_optimized_3d': rmsd_optimized_3d,
        'RMSD_2d': rmsd_2d
    })

    return rmsd_df

if __name__ == '__main__':
    dataset = 'gdb8'

    # Paths to SDF files
    precise_path = f'data/from-smiles/{dataset}-accurate/{dataset}.sdf'
    rdkit_3d_path = f'data/from-smiles/{dataset}-3d/{dataset}.sdf'
    optimized_3d_path = f'data/from-smiles/{dataset}-3d-opt/{dataset}.sdf'
    structure_2d_path = f'data/from-smiles/{dataset}-2d/{dataset}.sdf'

    # Process the molecules and calculate RMSD
    rmsd_df = process_molecules(dataset, precise_path, rdkit_3d_path, optimized_3d_path, structure_2d_path)

    # Save the results to a CSV file
    logging.info("Saving results to a CSV file...")
    rmsd_df.to_csv(f'data/rmsd-{dataset}.csv', index=False)

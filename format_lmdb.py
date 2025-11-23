import os
import argparse
from py_scripts.write_dude_multi import mol_parser, pocket_parser, write_lmdb
from openbabel import openbabel

def write_custom_lmdb(args, mol_data_path):
    mol2_path = ""
    pdb_path = ""
    for file in os.listdir(os.path.join(mol_data_path, args.name)):
        if file.endswith(".mol2"):
            mol2_path = file
        if file.endswith(".pdb"):
            pdb_path = file
    
    os.rename(os.path.join(mol_data_path, args.name, mol2_path), os.path.join(mol_data_path, args.name, "crystal_ligand.mol2"))

    protein_path = os.path.join(mol_data_path, args.name, pdb_path)
    mol_path = os.path.join(mol_data_path, args.name, 'molecules.smi')

    make_molecules: bool = not os.path.isfile(protein_path.replace(pdb_path, 'mols.lmdb')) or (args.force and args.molecules)
    make_pocket: bool = not os.path.isfile(protein_path.replace(pdb_path, 'pocket.lmdb')) or (args.force and args.pocket)

    if make_molecules:
        print('Parsing smiles to .lmdb')
        d_mol = (mol_parser(mol_path, "", 1))
        with open(protein_path.replace(pdb_path, 'real_mols.smi'), 'wt') as file:
            file.write('\n'.join([m['smi'] for m in d_mol]))
        write_lmdb(d_mol, protein_path.replace(pdb_path, 'mols.lmdb'))

    if make_pocket:
        print('Parsing pocket to .lmdb')
        d = pocket_parser(protein_path, protein_path.replace(pdb_path, 'crystal_ligand.mol2'), 0, args.raid)
        write_lmdb([d], protein_path.replace(pdb_path, 'pocket.lmdb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Writes a list of smiles and a protein pocket to lmdb files')
    parser.add_argument('name', metavar='NAME', type=str, help='the name of the protein')
    parser.add_argument('-d', '--data', type=str, help='the data directory', default=os.path.join(os.getcwd(), "data", "custom"))
    parser.add_argument('-f', '--force', action='store_true', help='force the program to overwrite an existing lmdb')
    parser.add_argument('-m', '--molecules', action='store_true', help='make molecules.lmdb')
    parser.add_argument('-p', '--pocket', action='store_true', help='make pocket.lmdb')
    parser.add_argument('-r', '--raid', type=int, help='raid', default=6)

    args = parser.parse_args()
    write_custom_lmdb(args, args.data)
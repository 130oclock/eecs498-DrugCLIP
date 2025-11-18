import os
import argparse
from py_scripts.write_dude_multi import mol_parser, write_lmdb, read_pdb

def all_pocket_parser(protein_path, pocket_index):
    protein = read_pdb(protein_path)
    pocket_atom_type = protein['atom_type']
    pocket_coord = protein['coord']
    pocket_residue_type = protein['residue_type']
    pocket_name = protein_path.split('/')[-2]
    return {'pocket': pocket_name,
            'pocket_index': pocket_index,
            "pocket_atoms": pocket_atom_type, 
            "pocket_coordinates": pocket_coord}

def write_custom_lmdb(args, mol_data_path):
    protein_path = os.path.join(mol_data_path, args.name, 'receptor.pdb')
    mol_path = os.path.join(mol_data_path, args.name, 'molecules.smi')

    make_molecules: bool = not os.path.isfile(protein_path.replace('receptor.pdb', 'mols.lmdb')) or (args.force and args.molecules)
    make_pocket: bool = not os.path.isfile(protein_path.replace('receptor.pdb', 'pocket.lmdb')) or (args.force and args.pocket)

    if make_molecules:
        print('Parsing smiles to .lmdb')
        d_mol = (mol_parser(mol_path, "", 1))
        with open(protein_path.replace('receptor.pdb', 'real_mols.smi'), 'wt') as file:
            file.write('\n'.join([m['smi'] for m in d_mol]))
        write_lmdb(d_mol, protein_path.replace('receptor.pdb', 'mols.lmdb'))

    if make_pocket:
        print('Parsing pocket to .lmdb')
        d = all_pocket_parser(protein_path, 0)
        write_lmdb([d], protein_path.replace('receptor.pdb', 'pocket.lmdb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Writes a list of smiles and a protein pocket to lmdb files')
    parser.add_argument('name', metavar='NAME', type=str, help='the name of the protein')
    parser.add_argument('-d', '--data', type=str, help='the data directory', default='./data/custom/')
    parser.add_argument('-f', '--force', action='store_true', help='force the program to overwrite an existing lmdb')
    parser.add_argument('-m', '--molecules', action='store_true', help='make molecules.lmdb')
    parser.add_argument('-p', '--pocket', action='store_true', help='make pocket.lmdb')

    args = parser.parse_args()
    write_custom_lmdb(args, args.data)
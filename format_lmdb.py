import os
import argparse
from py_scripts.write_dude_multi import mol_parser, pocket_parser, write_lmdb
from openbabel import openbabel


def write_custom_lmdb(args, mol_data_path):
    protein_name = f'{args.name}.pdb'

    pocket = os.path.join(mol_data_path, args.name, protein_name)
    mol_path = os.path.join(mol_data_path, args.name, 'molecules.smi')

    make_molecules: bool = not os.path.isfile(pocket.replace(protein_name, 'mols.lmdb')) or (args.force and args.molecules)
    make_pocket: bool = not os.path.isfile(pocket.replace(protein_name, 'pocket.lmdb')) or (args.force and args.pocket)

    if not os.path.isfile(pocket.replace(protein_name, 'crystal_ligand.mol2')):
        # converts the receptor pdb into mol2
        print('Converting receptor.pdb to crystal_ligand.mol2')
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "mol2")

        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, pocket)

        mol.AddHydrogens()
        print("Atoms:", mol.NumAtoms())

        obConversion.WriteFile(mol, pocket.replace(protein_name, 'crystal_ligand.mol2'))

    if make_molecules:
        print('Parsing smiles to .lmdb')
        d_mol = (mol_parser(mol_path, "", 1))
        write_lmdb(d_mol, pocket.replace(protein_name, 'mols.lmdb'))

    if make_pocket:
        print('Parsing pocket to .lmdb')
        d = pocket_parser(pocket, pocket.replace(protein_name, 'crystal_ligand.mol2'), 0)
        write_lmdb([d], pocket.replace(protein_name, 'pocket.lmdb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Writes a list of smiles and a protein pocket to lmdb files')
    parser.add_argument('name', metavar='NAME', type=str, help='the name of the protein')
    parser.add_argument('-d', '--data', type=str, help='the data directory', default='./data/custom/')
    parser.add_argument('-f', '--force', action='store_true', help='force the program to overwrite an existing lmdb')
    parser.add_argument('-l', '--ligand', action='store_true', help='make crystal_ligand.mol2')
    parser.add_argument('-m', '--molecules', action='store_true', help='make molecules.lmdb')
    parser.add_argument('-p', '--pocket', action='store_true', help='make pocket.lmdb')

    args = parser.parse_args()
    write_custom_lmdb(args, args.data)
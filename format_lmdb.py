import os
import argparse
from py_scripts.write_dude_multi import mol_parser, pocket_parser, write_lmdb
from openbabel import openbabel


def write_custom_lmdb(args, mol_data_path):
    pocket = os.path.join(mol_data_path, args.name, 'receptor.pdb')
    mol_path = os.path.join(mol_data_path, args.name, 'molecules.smi')

    # converts the receptor pdb into mol2
    print('Converting receptor.pdb to crystal_ligand.mol2')
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "mol2")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, pocket)

    mol.AddHydrogens()

    obConversion.WriteFile(mol, pocket.replace('receptor.pdb', 'crystal_ligand.mol2'))

    print('Parsing smiles to .lmdb')
    # format smiles into lmdb
    data = []
    d_mol = (mol_parser(mol_path, pocket.replace('receptor.pdb', 'crystal_ligand.mol2'), 1))

    data.extend(d_mol)
    write_lmdb(data, pocket.replace('receptor.pdb', 'mols.lmdb'))

    print('Parsing pocket to .lmdb')
    # write pocket into lmdb
    d = pocket_parser(pocket, pocket.replace('receptor.pdb', 'crystal_ligand.mol2'), 0)
    write_lmdb([d], pocket.replace('receptor.pdb', 'pocket.lmdb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='LMDB formatter',
        description='Writes a list of smiles and a protein pocket to lmdb files')
    parser.add_argument('name', metavar='N', type=str, help='The protein name')

    args = parser.parse_args()
    write_custom_lmdb(args, './data/custom/')
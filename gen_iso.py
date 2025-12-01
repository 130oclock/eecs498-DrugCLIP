from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import numpy as np

'''https://www.rdkit.org/docs/source/rdkit.Chem.EnumerateStereoisomers.html'''
#num_iter controls the number of stereoisomers generated (default 5)
#can write stereoisomers to a .txt file if file is not None. Generated file will be in same directory
def gen_isomers(mol, num_iter = 5, file = None):
    #Set enumeration options
    #opts = StereoEnumerationOptions(tryEmbedding=True)  # tryEmbedding=True ensures valid 3D configurations
    opts = StereoEnumerationOptions(
        tryEmbedding=True,
        maxIsomers=num_iter, 
        unique=True,
        onlyUnassigned=True
    )
    #Generate stereoisomers
    isomers = tuple(EnumerateStereoisomers(mol, options=opts))

    #Display results
    for i, iso in enumerate(isomers, 1):
        if not file == None:
            print(f"{Chem.MolToSmiles(iso, isomericSmiles=True)}",
              file=file)
        else:
            print(f"Isomer {i}: {Chem.MolToSmiles(iso, isomericSmiles=True)}")
        ##uncomment these two lines if you want to see each stereoisomer as a .png 
        ##click on generated .png and press enter to view next stereoisomer
        #visualize_mol(iso)
        #input()
        num_iter -= 1
        if num_iter <= 0:
            break

#takes a molecule as input and generates a picture of that molecule in a .png with file_name
def visualize_mol(mol, file_name = "atom.png"):
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(file_name)

if __name__ == '__main__':
    #insert SMILES string of molecule WITHOUT stereochmistry
    test_mol = Chem.MolFromSmiles('C=CCC1C=C(C)CC(C)CC(OC)C2OC(O)(C(=O)C(=O)N3CCCCC3C(=O)OC(C(C)=CC3CCC(O)C(OC)C3)C(C)C(O)CC1=O)C(C)CC2OC')
    visualize_mol(test_mol, file_name="original.png")
    with open("isomers_test.txt", "w") as f:
        gen_isomers(test_mol, file=f)

#SMILES string of Tacrolimus
#C=CC[C@@H]1/C=C(\C)C[C@H](C)C[C@H](OC)[C@H]2O[C@@](O)(C(=O)C(=O)N3CCCC[C@H]3C(=O)O[C@H](/C(C)=C/[C@@H]3CC[C@@H](O)[C@H](OC)C3)[C@H](C)[C@@H](O)CC1=O)[C@H](C)C[C@@H]2OC

#SMILES string of Tacrolimus without stereochemistry
#C=CCC1C=C(C)CC(C)CC(OC)C2OC(O)(C(=O)C(=O)N3CCCCC3C(=O)OC(C(C)=CC3CCC(O)C(OC)C3)C(C)C(O)CC1=O)C(C)CC2OC

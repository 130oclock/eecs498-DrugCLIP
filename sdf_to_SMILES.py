from rdkit import Chem
from gen_iso import visualize_mol, gen_isomers

##Writes num_iter of stereoisomers of molecule in sdf file to a .txt file which can be specified
with open("isomers.txt", "w") as f:
    ##paste in sdf file of molecule. sdf files can be downloaded from pubchem.
    suppl = Chem.SDMolSupplier("Structure2D_COMPOUND_CID_445643.sdf")
    for mol in suppl:
        if mol is not None:
            print("With stereo: ", Chem.MolToSmiles(mol))
            Chem.RemoveStereochemistry(mol) #strip stereochem
            print("No stereo: ", Chem.MolToSmiles(mol))
            #visualize_mol(mol) #optional
            gen_isomers(mol, num_iter= 6, file = f) #generate isomers 


import os
import argparse
import pandas as pd

def main(args):
    filenames = ['results-20251127-190857.csv']

    with open(os.path.join(args.data, args.name, "real_mols.smi"), "r") as file:
        lines = file.read().splitlines()
        
    scores_df = pd.DataFrame({
        'name': lines
    })

    for file in filenames:
        print("Reading", file)
        results_df = pd.read_csv(os.path.join(args.results, file))
        results_df.columns = ['name', 'score_' + file]

        scores_df = pd.merge(scores_df, results_df, on='name', how='outer')

    scores_df['mean'] = scores_df.mean(axis=1, numeric_only=True)
    print(scores_df)

    scores_df[['name', 'mean']].to_csv(os.path.join(args.results, "averaged.csv"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Writes a list of smiles and a protein pocket to lmdb files')
    parser.add_argument('name', metavar='NAME', type=str, help='the name of the protein')
    parser.add_argument('-d', '--data', type=str, help='the data directory', default=os.path.join(os.getcwd(), "data", "custom"))
    parser.add_argument('-r', '--results', type=str, help='the data directory', default=os.path.join(os.getcwd(), "test"))

    args = parser.parse_args()
    main(args)
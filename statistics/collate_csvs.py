import pandas as pd
import sys

if __name__ == '__main__':
    in_files = sys.argv[2:]
    out_file = sys.argv[1]

    dfs = [pd.read_csv(fname) for fname in in_files]
    final_df = pd.concat(dfs, ignore_index=True, copy=False)

    final_df.to_csv(out_file, index=False)

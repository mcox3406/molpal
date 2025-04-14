import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from molpal.models import mpn
from molpal.models.mpnmodels import MPNModel, MPNDropoutModel, MPNTwoOutputModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True,
                        help='Path to the MolPAL checkpoint directory containing model.pt and state.json')
    parser.add_argument('--input_csv', required=True,
                        help='Path to the input CSV file containing a "SMILES" column')
    parser.add_argument('--output_csv', required=True,
                        help='Path to save the output CSV file with SMILES and predictions')
    parser.add_argument('--conf_method', default='none', choices=['none', 'dropout', 'mve', 'twooutput'],
                        help='Confidence estimation method used during training (determines model type)')
    parser.add_argument('--dataset_type', default='regression', choices=['regression', 'classification'],
                        help='Type of dataset the model was trained for')
    parser.add_argument('--ncpu', type=int, default=1,
                        help='Number of CPU cores used for underlying model ops (esp. Ray)')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32],
                        help='Floating point precision used by the model')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    state_path = checkpoint_dir / 'state.json'
    model_path = checkpoint_dir / 'model.pt'

    if not state_path.exists() or not model_path.exists():
        print("Error: Checkpoint directory must contain 'state.json' and 'model.pt'.", file=sys.stderr)
        sys.exit(1)

    # 1. Load state information
    with open(state_path, 'r') as f:
        state = json.load(f)

    # 2. Instantiate the appropriate MPN model
    # pretty hacky for now
    print(f"Instantiating model (type: {args.conf_method}, dataset: {args.dataset_type})...")
    # pass required args -- assuming defaults for others for now.
    model_instance = mpn(
        conf_method=args.conf_method,
        dataset_type=args.dataset_type,
        ncpu=args.ncpu,
        precision=args.precision
    )

    # 3. Load the model state and scaler
    print(f"Loading model state from: {state_path}")
    model_instance.load(str(state_path))

    # 4. Load SMILES from input CSV
    print(f"Loading SMILES from: {args.input_csv}")
    try:
        df_input = pd.read_csv(args.input_csv)
        df_input.columns = df_input.columns.str.upper()
        if 'SMILES' not in df_input.columns:
            raise ValueError('Input CSV must contain a "SMILES" column.')
        smiles_list = df_input['SMILES'].tolist() or df_input['smiles'].tolist()
    except Exception as e:
        print(f"Error loading input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Perform prediction
    print(f"Predicting properties for {len(smiles_list)} molecules...")
    if args.conf_method == 'none':
        if not isinstance(model_instance, MPNModel):
             raise TypeError("Model type mismatch for conf_method 'none'")
        means = model_instance.get_means(smiles_list)
        variances = None
    elif args.conf_method in ('dropout', 'mve', 'twooutput'):
        if not isinstance(model_instance, (MPNDropoutModel, MPNTwoOutputModel)):
             raise TypeError(f"Model type mismatch for conf_method '{args.conf_method}'")
        means, variances = model_instance.get_means_and_vars(smiles_list)
    else:
         # should be caught by argparse choices
         raise ValueError(f"Unsupported conf_method: {args.conf_method}")

    # 6. Format and save output
    print("Formatting output...")
    df_output = pd.DataFrame({'SMILES': smiles_list})
    # assuming single-task prediction
    df_output['mean'] = means.flatten()
    if variances is not None:
        df_output['variance'] = variances.flatten()

    print(f"Saving predictions to: {args.output_csv}")
    df_output.to_csv(args.output_csv, index=False)

    print("Prediction complete.")

if __name__ == '__main__':
    main() 
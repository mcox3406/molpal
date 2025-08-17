import os
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Iterable, Optional
import subprocess
import json

import numpy as np
from configargparse import ArgumentParser

from molpal.objectives.base import Objective


class BoltzObjective(Objective):
    """A BoltzObjective calculates the objective function by calculating the
    binding affinity using Boltz-2

    Attributes
    ----------
    c : int
        the min/maximization constant, depending on the objective
    receptor_path : str
        path to the receptor PDB file
    use_msa_server : bool
        whether to use MSA server for predictions
    output_dir : Path
        directory for storing Boltz outputs
    temp_dir : Path
        temporary directory for input YAML files

    Parameters
    ----------
    receptor_path : str
        path to the receptor PDB file for binding affinity prediction
    use_msa_server : bool, default=True
        whether to use MSA server for predictions
    output_dir : str, default="boltz_outputs"
        directory to store Boltz prediction outputs
    minimize : bool, default=True
        whether this objective should be minimized (True for binding affinity)
    **kwargs
        additional and unused keyword arguments
    """

    def __init__(
        self,
        objective_config: str,
        minimize: bool = True,
        **kwargs,
    ):
        receptor_path, use_msa_server, output_dir = parse_config(objective_config)
        
        if not os.path.exists(receptor_path):
            raise FileNotFoundError(f"Receptor file not found: {receptor_path}")
        
        self.receptor_path = os.path.abspath(receptor_path)
        self.use_msa_server = use_msa_server
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="boltz_"))
        
        super().__init__(minimize=minimize)

    def _create_input_yaml(self, smi: str, output_name: str) -> str:
        """Create a Boltz input YAML file for a SMILES string"""
        yaml_content = {
            "target": {
                "sequences": [
                    {"protein": {"id": "receptor", "path": self.receptor_path}}
                ]
            },
            "ligands": [
                {"smiles": smi, "id": "ligand"}
            ],
            "outputs": {
                "affinity": True
            }
        }
        
        yaml_path = self.temp_dir / f"{output_name}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        return str(yaml_path)

    def _run_boltz_prediction(self, yaml_path: str, output_name: str) -> Optional[float]:
        """Run Boltz prediction for a single YAML input"""
        try:
            cmd = ["boltz", "predict", yaml_path]
            if self.use_msa_server:
                cmd.append("--use_msa_server")
            
            # Set output directory for this prediction
            prediction_output_dir = self.output_dir / output_name
            prediction_output_dir.mkdir(exist_ok=True)
            
            # Run Boltz prediction
            result = subprocess.run(
                cmd,
                cwd=str(prediction_output_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per prediction
            )
            
            if result.returncode != 0:
                print(f"Boltz prediction failed for {output_name}: {result.stderr}")
                return None
            
            # Parse the affinity output
            return self._parse_affinity_output(prediction_output_dir)
            
        except subprocess.TimeoutExpired:
            print(f"Boltz prediction timed out for {output_name}")
            return None
        except Exception as e:
            print(f"Error running Boltz prediction for {output_name}: {e}")
            return None

    def _parse_affinity_output(self, output_dir: Path) -> Optional[float]:
        """Parse the binding affinity from Boltz output files"""
        try:
            # Look for JSON output files that contain affinity predictions
            json_files = list(output_dir.glob("*.json"))
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract affinity prediction value
                # Based on Boltz-2 readme: use affinity_pred_value for ligand optimization
                if 'affinity_pred_value' in data:
                    affinity = data['affinity_pred_value']
                    # Convert to numeric if it's a string
                    if isinstance(affinity, str):
                        try:
                            affinity = float(affinity)
                        except ValueError:
                            continue
                    return float(affinity)
                
                # Fallback to affinity_probability_binary if pred_value not available
                elif 'affinity_probability_binary' in data:
                    prob = data['affinity_probability_binary']
                    if isinstance(prob, str):
                        try:
                            prob = float(prob)
                        except ValueError:
                            continue
                    # Convert probability to a score (higher probability = lower score for minimization)
                    return -np.log(max(prob, 1e-8))  # Avoid log(0)
            
            print(f"No affinity data found in output directory: {output_dir}")
            return None
            
        except Exception as e:
            print(f"Error parsing affinity output from {output_dir}: {e}")
            return None

    def forward(self, smis: Iterable[str], **kwargs) -> Dict[str, Optional[float]]:
        """Calculate the binding affinities for a list of SMILES strings

        Parameters
        ----------
        smis : Iterable[str]
            the SMILES strings of the molecules to evaluate
        **kwargs
            additional and unused positional and keyword arguments

        Returns
        -------
        scores : Dict[str, Optional[float]]
            a map from SMILES string to binding affinity score. Molecules that failed
            to predict will be scored as None
        """
        results = {}
        
        for i, smi in enumerate(smis):
            output_name = f"pred_{i}"
            
            # Create input YAML file
            yaml_path = self._create_input_yaml(smi, output_name)
            
            # Run Boltz prediction
            affinity = self._run_boltz_prediction(yaml_path, output_name)
            
            # Apply minimization constant and store result
            if affinity is not None:
                results[smi] = self.c * affinity
            else:
                results[smi] = None
        
        return results

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def parse_config(config: str):
    """Parse a BoltzObjective configuration file

    Parameters
    ----------
    config : str
        the config file to parse

    Returns
    -------
    receptor_path : str
        the filepath of the receptor PDB file
    use_msa_server : bool
        whether to use MSA server for predictions
    output_dir : str
        directory to store Boltz prediction outputs
    """
    parser = ArgumentParser()
    parser.add_argument("config", is_config_file=True)
    parser.add_argument("--receptor-path", required=True, 
                       help="Path to the receptor PDB file")
    parser.add_argument("--use-msa-server", action="store_true", default=True,
                       help="Use MSA server for predictions")
    parser.add_argument("--output-dir", default="boltz_outputs",
                       help="Directory to store Boltz outputs")

    args = parser.parse_args([config])
    return args.receptor_path, args.use_msa_server, args.output_dir
#!/usr/bin/env python3
"""
Test script for Boltz-2 integration with MolPAL

This script tests the basic functionality of the Boltz objective without
requiring actual Boltz-2 installation (mocks the subprocess calls).
"""

import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from molpal.objectives import objective


def create_test_config():
    """Create a test configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write('receptor-path = /tmp/test_receptor.pdb\n')
        f.write('use-msa-server = false\n')
        f.write('output-dir = /tmp/boltz_test\n')
        return f.name


def create_dummy_receptor():
    """Create a dummy PDB file"""
    with open('/tmp/test_receptor.pdb', 'w') as f:
        f.write('ATOM      1  CA  ALA A   1      20.000  20.000  20.000  1.00 10.00           C\n')


def create_mock_output(output_dir, affinity_value=-8.5):
    """Create mock Boltz output files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mock JSON output with affinity prediction
    output_file = os.path.join(output_dir, "prediction_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "affinity_pred_value": affinity_value,
            "affinity_probability_binary": 0.85
        }, f)


def test_boltz_objective():
    """Test the Boltz objective functionality"""
    config_path = None
    try:
        # Setup test files
        config_path = create_test_config()
        create_dummy_receptor()
        
        # Create Boltz objective
        boltz_obj = objective('boltz', config_path)
        print("Boltz objective created successfully")
        
        # Mock the subprocess call to avoid requiring actual Boltz installation
        test_smiles = ["CCO", "CC(=O)O"]  # ethanol and acetic acid
        
        with patch('subprocess.run') as mock_run:
            # Mock successful subprocess call
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            # Mock the output parsing by creating fake output files
            original_parse = boltz_obj._parse_affinity_output
            def mock_parse_affinity_output(output_dir):
                # Create mock output for testing
                create_mock_output(str(output_dir), affinity_value=-7.2)
                return original_parse(output_dir)
            
            boltz_obj._parse_affinity_output = mock_parse_affinity_output
            
            # Test prediction
            results = boltz_obj.forward(test_smiles)
            
            print(f"Prediction completed successfully")
            print(f"Results: {results}")
            
            # Verify results
            assert len(results) == len(test_smiles), "Should return results for all SMILES"
            for smi in test_smiles:
                assert smi in results, f"Missing result for {smi}"
                assert results[smi] is not None, f"Result should not be None for {smi}"
                
            print("All assertions passed")
        
        # Test cleanup
        boltz_obj.cleanup()
        print("Cleanup completed successfully")
        
        print("\nAll tests passed! Boltz-2 integration is working correctly.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise
    finally:
        # Cleanup test files
        if config_path and os.path.exists(config_path):
            os.unlink(config_path)
        if os.path.exists('/tmp/test_receptor.pdb'):
            os.unlink('/tmp/test_receptor.pdb')


if __name__ == "__main__":
    test_boltz_objective()
"""Utility to verify alpha structures and their documentation."""
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def verify_alpha_documentation(results_dir: str = 'evaluation_results') -> List[Dict]:
    """
    Verify that alpha structures match across all documentation.
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        List of discrepancies found
    """
    discrepancies = []
    
    # Check experiment_results.csv
    if os.path.exists('experiment_results.csv'):
        df = pd.read_csv('experiment_results.csv')
        
        # For each alpha in experiment results
        for idx, row in df.iterrows():
            alpha_str = row['alpha_structure']
            
            # Check if corresponding evaluation directory exists
            eval_dir = os.path.join(results_dir, alpha_str)
            if not os.path.exists(eval_dir):
                discrepancies.append({
                    'type': 'missing_evaluation',
                    'alpha': alpha_str,
                    'location': 'experiment_results.csv',
                    'row': idx
                })
                continue
                
            # Check alpha structure in evaluation directory
            structure_file = os.path.join(eval_dir, 'alpha_structure.txt')
            if os.path.exists(structure_file):
                with open(structure_file, 'r') as f:
                    content = f.read()
                    if alpha_str not in content:
                        discrepancies.append({
                            'type': 'structure_mismatch',
                            'experiment_alpha': alpha_str,
                            'evaluation_alpha': content.split('\n')[0],
                            'file': structure_file
                        })
    
    return discrepancies

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run verification
    discrepancies = verify_alpha_documentation()
    
    if discrepancies:
        print("\nDiscrepancies found:")
        for d in discrepancies:
            print(f"\nType: {d['type']}")
            if d['type'] == 'missing_evaluation':
                print(f"Alpha: {d['alpha']}")
                print(f"Missing evaluation in: {d['location']}")
            elif d['type'] == 'structure_mismatch':
                print(f"Experiment alpha: {d['experiment_alpha']}")
                print(f"Evaluation alpha: {d['evaluation_alpha']}")
                print(f"File: {d['file']}")
    else:
        print("\nNo discrepancies found. All alpha structures match across documentation.") 
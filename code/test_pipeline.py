"""
Test the main pipeline on sample claims before running on full dataset.
"""

import pandas as pd
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import process_claim, load_user_history, load_evidence_requirements

def test_on_sample_claims(num_samples=3):
    """Test pipeline on sample claims."""
    print("Loading sample claims...")
    sample_df = pd.read_csv('../dataset/sample_claims.csv')
    
    print(f"Testing on {num_samples} sample claims...\n")
    
    user_history_df = load_user_history()
    evidence_df = load_evidence_requirements()
    
    for idx, row in sample_df.head(num_samples).iterrows():
        print(f"{'='*60}")
        print(f"Test {idx + 1}: User {row['user_id']}, Object: {row['claim_object']}")
        print(f"Expected: issue_type={row['issue_type']}, object_part={row['object_part']}, status={row['claim_status']}")
        print(f"{'='*60}")
        
        try:
            result = process_claim(row, user_history_df, evidence_df)
            
            print(f"\nPipeline Result:")
            print(f"  issue_type: {result['issue_type']}")
            print(f"  object_part: {result['object_part']}")
            print(f"  claim_status: {result['claim_status']}")
            print(f"  evidence_standard_met: {result['evidence_standard_met']}")
            print(f"  risk_flags: {result['risk_flags']}")
            print(f"  severity: {result['severity']}")
            
            # Compare
            if result['issue_type'] == row['issue_type']:
                print("  ✓ Issue type matches")
            else:
                print(f"  ✗ Issue type mismatch (expected: {row['issue_type']})")
            
            if result['object_part'] == row['object_part']:
                print("  ✓ Object part matches")
            else:
                print(f"  ✗ Object part mismatch (expected: {row['object_part']})")
            
            if result['claim_status'] == row['claim_status']:
                print("  ✓ Claim status matches")
            else:
                print(f"  ✗ Claim status mismatch (expected: {row['claim_status']})")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    test_on_sample_claims(num_samples=3)

"""
Evaluation script for multi-modal evidence review system.
Evaluates predictions against ground truth on sample_claims.csv.
"""

import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_sample_claims():
    """Load sample claims with ground truth."""
    df = pd.read_csv('../../dataset/sample_claims.csv')
    return df

def load_predictions():
    """Load predictions from output.csv."""
    df = pd.read_csv('../../output.csv')
    return df

def calculate_metrics(predictions_df, sample_df):
    """Calculate evaluation metrics."""
    # Merge predictions with ground truth
    merged = predictions_df.merge(
        sample_df[['user_id', 'claim_status', 'issue_type', 'object_part', 'severity']],
        on='user_id',
        suffixes=('_pred', '_true')
    )
    
    metrics = {
        'total_claims': len(merged),
        'claim_status_accuracy': (merged['claim_status_pred'] == merged['claim_status_true']).mean(),
        'issue_type_accuracy': (merged['issue_type_pred'] == merged['issue_type_true']).mean(),
        'object_part_accuracy': (merged['object_part_pred'] == merged['object_part_true']).mean(),
        'severity_accuracy': (merged['severity_pred'] == merged['severity_true']).mean()
    }
    
    # Per-status breakdown
    status_breakdown = {}
    for status in ['supported', 'contradicted', 'not_enough_information']:
        status_mask = merged['claim_status_true'] == status
        if status_mask.sum() > 0:
            status_accuracy = (merged[status_mask]['claim_status_pred'] == merged[status_mask]['claim_status_true']).mean()
            status_breakdown[status] = {
                'count': status_mask.sum(),
                'accuracy': status_accuracy
            }
    
    metrics['status_breakdown'] = status_breakdown
    
    return metrics, merged

def print_metrics(metrics):
    """Print evaluation metrics."""
    print("="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"\nTotal Claims: {metrics['total_claims']}")
    print(f"\nOverall Accuracy:")
    print(f"  Claim Status: {metrics['claim_status_accuracy']:.2%}")
    print(f"  Issue Type: {metrics['issue_type_accuracy']:.2%}")
    print(f"  Object Part: {metrics['object_part_accuracy']:.2%}")
    print(f"  Severity: {metrics['severity_accuracy']:.2%}")
    
    print(f"\nPer-Status Breakdown:")
    for status, data in metrics['status_breakdown'].items():
        print(f"  {status}: {data['count']} claims, {data['accuracy']:.2%} accuracy")

def print_misclassifications(merged_df):
    """Print misclassified examples."""
    misclassified = merged_df[merged_df['claim_status_pred'] != merged_df['claim_status_true']]
    
    if len(misclassified) > 0:
        print(f"\n{'='*60}")
        print(f"MISCLASSIFICATIONS ({len(misclassified)} claims)")
        print("="*60)
        
        for _, row in misclassified.iterrows():
            print(f"\nUser: {row['user_id']}")
            print(f"  Object: {row['claim_object']}")
            print(f"  Predicted: {row['claim_status_pred']}")
            print(f"  Actual: {row['claim_status_true']}")
            print(f"  Issue Type: {row['issue_type_pred']} vs {row['issue_type_true']}")
            print(f"  Object Part: {row['object_part_pred']} vs {row['object_part_true']}")

def main():
    """Main evaluation function."""
    print("Loading data...")
    
    try:
        sample_df = load_sample_claims()
        predictions_df = load_predictions()
        
        print(f"Sample claims: {len(sample_df)}")
        print(f"Predictions: {len(predictions_df)}")
        
        # Calculate metrics
        metrics, merged_df = calculate_metrics(predictions_df, sample_df)
        
        # Print results
        print_metrics(metrics)
        print_misclassifications(merged_df)
        
        print(f"\n{'='*60}")
        print("Evaluation complete!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure output.csv exists in the repository root")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

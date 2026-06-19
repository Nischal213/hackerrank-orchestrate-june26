import os
import json
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import pandas as pd
from vlm_prompts import SINGLE_IMAGE_ANALYSIS_PROMPT, MULTI_IMAGE_ANALYSIS_PROMPT

def load_sample_claims():
    """Load sample claims for testing."""
    df = pd.read_csv('../dataset/sample_claims.csv')
    return df

def analyze_single_image(model, image_path, claim_object, user_claim):
    """Analyze a single image using VLM."""
    img = Image.open(image_path)
    
    prompt = SINGLE_IMAGE_ANALYSIS_PROMPT.format(
        claim_object=claim_object,
        user_claim=user_claim
    )
    
    try:
        response = model.generate_content([prompt, img])
        # Try to parse JSON from response
        response_text = response.text.strip()
        
        # Extract JSON if it's wrapped in markdown code blocks
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response.text}")
        return None
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def test_sample_claims(model, num_samples=5):
    """Test VLM on sample claims."""
    df = load_sample_claims()
    
    print(f"\nTesting VLM on {num_samples} sample claims...\n")
    
    for idx, row in df.head(num_samples).iterrows():
        user_id = row['user_id']
        image_paths = row['image_paths'].split(';')
        user_claim = row['user_claim']
        claim_object = row['claim_object']
        
        print(f"{'='*60}")
        print(f"Test {idx + 1}: User {user_id}, Object: {claim_object}")
        print(f"Images: {len(image_paths)} image(s)")
        print(f"{'='*60}")
        
        # Test first image only for now
        first_image_path = f"../dataset/{image_paths[0]}"
        
        if not os.path.exists(first_image_path):
            print(f"ERROR: Image not found: {first_image_path}")
            continue
        
        result = analyze_single_image(model, first_image_path, claim_object, user_claim)
        
        if result:
            print("\nVLM Analysis Result:")
            print(json.dumps(result, indent=2))
            
            # Compare with expected output
            expected = {
                'issue_type': row['issue_type'],
                'object_part': row['object_part'],
                'claim_status': row['claim_status'],
                'severity': row['severity']
            }
            
            print(f"\nExpected: issue_type={expected['issue_type']}, object_part={expected['object_part']}, claim_status={expected['claim_status']}")
            print(f"VLM detected: issue_type={result.get('issue_type')}, object_part={result.get('object_part_visible')}")
            
            # Simple comparison
            if result.get('issue_type') == expected['issue_type']:
                print("✓ Issue type matches")
            else:
                print("✗ Issue type mismatch")
            
            if result.get('object_part_visible') == expected['object_part']:
                print("✓ Object part matches")
            else:
                print("✗ Object part mismatch")
        
        print()

def main():
    """Main test function."""
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY_1')
    if not api_key:
        print("ERROR: GEMINI_API_KEY_1 not configured")
        sys.exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Use Gemini 2.5 Flash for testing (faster)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    print("Testing VLM analysis on sample claims...")
    test_sample_claims(model, num_samples=3)
    
    print("\n✓ VLM testing complete!")

if __name__ == "__main__":
    main()

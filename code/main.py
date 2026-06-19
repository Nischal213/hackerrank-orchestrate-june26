"""
Main processing pipeline for damage claim verification using VLM.
"""

import os
import json
import sys
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from PIL import Image
import pandas as pd
from typing import List, Dict, Optional
from vlm_prompts import SINGLE_IMAGE_ANALYSIS_PROMPT, MULTI_IMAGE_ANALYSIS_PROMPT, RISK_ASSESSMENT_PROMPT

# Load environment variables
load_dotenv()

# API key rotation
API_KEYS = [
    os.getenv('GEMINI_API_KEY_1'),
    os.getenv('GEMINI_API_KEY_2'),
    os.getenv('GEMINI_API_KEY_3')
]
API_KEYS = [k for k in API_KEYS if k and k != 'your_first_api_key_here' and k != 'your_second_api_key_here' and k != 'your_third_api_key_here']

if not API_KEYS:
    print("ERROR: No valid API keys found")
    sys.exit(1)

CURRENT_KEY_INDEX = 0

def get_model():
    """Get Gemini model with API key rotation."""
    global CURRENT_KEY_INDEX
    api_key = API_KEYS[CURRENT_KEY_INDEX]
    genai.configure(api_key=api_key)
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    return genai.GenerativeModel('gemini-2.5-flash-lite')

def call_with_retry(model, prompt, images=None, max_retries=3):
    """Call VLM with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            if images:
                response = model.generate_content([prompt] + images)
            else:
                response = model.generate_content(prompt)
            return response
        except (ResourceExhausted, Exception) as e:
            error_str = str(e).lower()
            
            # Robust check matching exception type, status code attribute, or substrings
            is_rate_limit = (
                isinstance(e, ResourceExhausted) or
                getattr(e, 'code', None) == 429 or
                '429' in error_str or
                'quota' in error_str
            )
            
            if is_rate_limit:
                wait_time = min(60, (2 ** attempt) * 10)  # 10s, 20s, 40s
                print(f"  Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                
                # Rotate to the next API key safely
                global CURRENT_KEY_INDEX
                if API_KEYS:
                    api_key = API_KEYS[CURRENT_KEY_INDEX]
                    genai.configure(api_key=api_key)
                    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
                    
                    # Instantiate using the recommended testing model
                    model = genai.GenerativeModel('gemini-2.5-flash-lite')
            else:
                # Raise other unexpected errors immediately
                raise e
                
    raise Exception("Max retries exceeded for API call")

def load_user_history():
    """Load user history data."""
    df = pd.read_csv('../dataset/user_history.csv')
    return df.set_index('user_id')

def load_evidence_requirements():
    """Load evidence requirements data."""
    df = pd.read_csv('../dataset/evidence_requirements.csv')
    return df

def parse_json_response(response_text: str) -> Optional[Dict]:
    """Parse JSON from VLM response, handling markdown code blocks."""
    response_text = response_text.strip()
    
    # Extract JSON if wrapped in markdown code blocks
    if '```json' in response_text:
        response_text = response_text.split('```json')[1].split('```')[0].strip()
    elif '```' in response_text:
        response_text = response_text.split('```')[1].split('```')[0].strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response_text}")
        return None

def analyze_single_image(model, image_path: str, claim_object: str, user_claim: str) -> Optional[Dict]:
    """Analyze a single image using VLM."""
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"  Cannot open image {image_path}: {e}")
        return None
    
    prompt = SINGLE_IMAGE_ANALYSIS_PROMPT.format(
        claim_object=claim_object,
        user_claim=user_claim
    )
    
    try:
        response = call_with_retry(model, prompt, [img])
        return parse_json_response(response.text)
    except Exception as e:
        print(f"Analysis error for {image_path}: {e}")
        return None

def analyze_multiple_images(model, image_paths: List[str], claim_object: str, user_claim: str) -> Optional[Dict]:
    """Analyze multiple images using VLM."""
    images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"  Cannot open image {path}: {e}")
    
    if not images:
        print(f"  No valid images could be opened")
        return None
    
    prompt = MULTI_IMAGE_ANALYSIS_PROMPT.format(
        claim_object=claim_object,
        user_claim=user_claim,
        num_images=len(images)
    )
    
    try:
        response = call_with_retry(model, prompt, images)
        return parse_json_response(response.text)
    except Exception as e:
        print(f"Multi-image analysis error: {e}")
        return None

def determine_claim_status(analysis: Dict, risk_flags: List[str]) -> str:
    """Determine final claim status based on analysis and risks."""
    if not analysis:
        return "not_enough_information"
    
    supports = analysis.get('supports_claim', False)
    contradicts = analysis.get('contradicts_claim', False)
    evidence_sufficient = analysis.get('evidence_sufficient', False)
    damage_visible = analysis.get('damage_visible', False)
    
    # Primary determination based on VLM analysis
    if contradicts:
        return "contradicted"
    elif evidence_sufficient and damage_visible:
        # If evidence is sufficient and damage is visible, support the claim
        # Only downgrade for severe manipulation risks
        if 'possible_manipulation' in risk_flags or 'non_original_image' in risk_flags:
            return "not_enough_information"
        return "supported"
    elif not evidence_sufficient:
        return "not_enough_information"
    else:
        return "not_enough_information"

def process_claim(row: pd.Series, user_history_df: pd.DataFrame, evidence_df: pd.DataFrame) -> Dict:
    """Process a single claim row, safely parsing both single and multi-image VLM responses."""
    user_id = row['user_id']
    image_paths_str = row['image_paths']
    user_claim = row['user_claim']
    claim_object = row['claim_object']
    
    # Parse image paths
    image_paths = [f"../dataset/{path.strip()}" for path in image_paths_str.split(';')]
    
    # Get user history
    user_history = user_history_df.loc[user_id].to_dict() if user_id in user_history_df.index else {}
    
    # Get VLM model
    model = get_model()
    
    # Analyze images
    if len(image_paths) == 1:
        analysis = analyze_single_image(model, image_paths[0], claim_object, user_claim)
    else:
        analysis = analyze_multiple_images(model, image_paths, claim_object, user_claim)
    
    if not analysis:
        # Fallback if analysis fails
        return {
            'user_id': user_id,
            'image_paths': image_paths_str,
            'user_claim': user_claim,
            'claim_object': claim_object,
            'evidence_standard_met': False,
            'evidence_standard_met_reason': 'Image analysis failed',
            'risk_flags': 'none',
            'issue_type': 'unknown',
            'object_part': 'unknown',
            'claim_status': 'not_enough_information',
            'claim_status_justification': 'Unable to analyze images',
            'supporting_image_ids': 'none',
            'valid_image': False,
            'severity': 'unknown'
        }
    
    # Extract key fields and flags depending on the JSON layout structure (Single vs Multi-Image)
    if 'overall_assessment' in analysis:
        # --- MULTI-IMAGE PARSING FIX ---
        oa = analysis['overall_assessment']
        evidence_sufficient = oa.get('evidence_sufficient', False)
        evidence_reason = oa.get('evidence_reason', '')
        
        # Format supporting image IDs safely from list to semi-colon separated string
        supporting_images = analysis.get('supporting_image_ids', ['none'])
        if isinstance(supporting_images, list):
            supporting_image_ids = ';'.join(supporting_images)
        else:
            supporting_image_ids = str(supporting_images)
            
        severity = analysis.get('severity', 'unknown')
        
        # Aggregate issue_type and object_part across ALL images by looking for concrete features
        issue_type = 'unknown'
        object_part = 'unknown'
        for img_res in analysis.get('per_image_analysis', []):
            img_issue = img_res.get('issue_type', 'none')
            if img_issue and img_issue not in ['none', 'unknown']:
                issue_type = img_issue
                object_part = img_res.get('object_part_visible', 'unknown')
                break
        
        # Fallback to the first element if no explicit issue type was highlighted
        if issue_type == 'unknown' and analysis.get('per_image_analysis'):
            issue_type = analysis['per_image_analysis'][0].get('issue_type', 'unknown')
            object_part = analysis['per_image_analysis'][0].get('object_part_visible', 'unknown')

        # Map final verification status securely out of nested assessment fields
        if oa.get('contradicts_claim', False):
            claim_status = "contradicted"
        elif oa.get('supports_claim', False) and (oa.get('damage_visible_in_any_image', False) or oa.get('evidence_sufficient', False)):
            claim_status = "supported"
        else:
            claim_status = "not_enough_information"
            
        # Multi-image structure collects risk flags as a direct root array
        model_risk_flags = analysis.get('risk_flags', [])
        if isinstance(model_risk_flags, list):
            risk_flags = [f for f in model_risk_flags if f != 'none']
        else:
            risk_flags = [model_risk_flags] if model_risk_flags != 'none' else []
            
        valid_image = 'blurry_image' not in risk_flags
    else:
        # --- SINGLE IMAGE PARSING (Kept from existing framework) ---
        issue_type = analysis.get('issue_type', 'unknown')
        object_part = analysis.get('object_part_visible', 'unknown')
        evidence_sufficient = analysis.get('evidence_sufficient', False)
        evidence_reason = analysis.get('evidence_reason', '')
        supporting_image_ids = 'img_1' if analysis.get('damage_visible') else 'none'
        severity = analysis.get('severity', 'unknown')
        
        # Pull risks out of individual quality object attributes
        image_quality = analysis.get('image_quality', {})
        risk_flags = []
        if image_quality.get('blurry', False): 
            risk_flags.append('blurry_image')
        if image_quality.get('low_light_or_glare', False): 
            risk_flags.append('low_light_or_glare')
        if image_quality.get('cropped_or_obstructed', False): 
            risk_flags.append('cropped_or_obstructed')
        if image_quality.get('wrong_angle', False): 
            risk_flags.append('wrong_angle')
            
        valid_image = image_quality.get('clear', True) and not image_quality.get('blurry', False)
        claim_status = determine_claim_status(analysis, risk_flags)
    
    # Common addition: Append historical warning flags across both execution models
    if user_history.get('history_flags') and 'none' not in user_history['history_flags']:
        if 'user_history_risk' not in risk_flags:
            risk_flags.append('user_history_risk')
            
    if not risk_flags:
        risk_flags = ['none']
    
    risk_flags_str = ';'.join(risk_flags)
    
    # Direct formatting override for multi-image vs single-image string justifications
    justification_parts = []
    if evidence_reason:
        justification_parts.append(evidence_reason)
        
    if 'overall_assessment' in analysis:
        descriptions = [img.get('damage_description') for img in analysis.get('per_image_analysis', []) if img.get('damage_description')]
        if descriptions:
            justification_parts.append(f"Visible details: {'; '.join(descriptions)}")
    elif analysis.get('damage_visible') and analysis.get('damage_description'):
        justification_parts.append(f"Visible damage: {analysis['damage_description']}")
        
    if risk_flags and 'none' not in risk_flags:
        justification_parts.append(f"Risk flags: {'; '.join(risk_flags)}")
        
    if user_history.get('history_flags') and 'none' not in user_history['history_flags']:
        justification_parts.append(f"User history: {user_history.get('history_summary', 'Concerning patterns identified')}")
        
    justification = '. '.join(justification_parts) if justification_parts else "Analysis completed."
    
    return {
        'user_id': user_id,
        'image_paths': image_paths_str,
        'user_claim': user_claim,
        'claim_object': claim_object,
        'evidence_standard_met': evidence_sufficient,
        'evidence_standard_met_reason': evidence_reason,
        'risk_flags': risk_flags_str,
        'issue_type': issue_type,
        'object_part': object_part,
        'claim_status': claim_status,
        'claim_status_justification': justification,
        'supporting_image_ids': supporting_image_ids,
        'valid_image': valid_image,
        'severity': severity
    }

def main():
    """Main processing function."""
    print("Loading data...")
    
    # Load input data
    claims_df = pd.read_csv('../dataset/claims.csv')
    user_history_df = load_user_history()
    evidence_df = load_evidence_requirements()
    
    print(f"Processing {len(claims_df)} claims...")
    
    # Process each claim
    results = []
    for idx, row in claims_df.iterrows():
        print(f"Processing claim {idx + 1}/{len(claims_df)}...")
        
        try:
            result = process_claim(row, user_history_df, evidence_df)
            results.append(result)
            # Add delay between claims to avoid rate limiting
            if idx < len(claims_df) - 1:
                time.sleep(2)
        except Exception as e:
            print(f"Error processing claim {idx + 1}: {e}")
            # Add fallback result
            results.append({
                'user_id': row['user_id'],
                'image_paths': row['image_paths'],
                'user_claim': row['user_claim'],
                'claim_object': row['claim_object'],
                'evidence_standard_met': False,
                'evidence_standard_met_reason': f'Processing error: {str(e)}',
                'risk_flags': 'none',
                'issue_type': 'unknown',
                'object_part': 'unknown',
                'claim_status': 'not_enough_information',
                'claim_status_justification': f'Error during processing: {str(e)}',
                'supporting_image_ids': 'none',
                'valid_image': False,
                'severity': 'unknown'
            })
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Ensure correct column order
    column_order = [
        'user_id', 'image_paths', 'user_claim', 'claim_object',
        'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
        'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
        'supporting_image_ids', 'valid_image', 'severity'
    ]
    output_df = output_df[column_order]
    
    # Save output
    output_df.to_csv('../output.csv', index=False)
    print(f"\n✓ Processing complete! Results saved to output.csv")
    print(f"Total claims processed: {len(results)}")

if __name__ == "__main__":
    main()
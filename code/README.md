# Multi-Modal Evidence Review System

A VLM-based system for verifying damage claims using images, claim conversations, user history, and evidence requirements.

## Overview

This system processes damage claims for three object types:
- **Cars**: Front/rear bumpers, doors, windshields, side mirrors, headlights, taillights, etc.
- **Laptops**: Screens, keyboards, trackpads, hinges, lids, corners, ports, etc.
- **Packages**: Boxes, corners, seals, labels, contents, items, etc.

## Architecture

The system uses:
- **Gemini 2.5 Flash Lite** for image analysis and text processing
- **Structured prompting** to extract claim information from conversations
- **Multi-modal analysis** combining images and text
- **API key rotation** to handle rate limits
- **Exponential backoff** for retry logic

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your Gemini API keys
```

## Usage

### Run on full dataset:
```bash
python main.py
```
This processes `dataset/claims.csv` and generates `output.csv`.

### Test on sample claims:
```bash
python test_pipeline.py
```
This tests the pipeline on a few sample claims from `dataset/sample_claims.csv`.

### Test Gemini connection:
```bash
python test_gemini.py
```

### Verify environment setup:
```bash
python setup_env.py
```

## File Structure

```
code/
├── main.py                    # Main processing pipeline
├── vlm_prompts.py             # VLM prompt templates
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in git)
├── .env.example               # Environment variable template
├── setup_env.py              # Environment setup verification
├── test_gemini.py            # Gemini API connection test
├── test_vlm_analysis.py      # VLM analysis testing
├── test_pipeline.py          # End-to-end pipeline testing
└── evaluation/
    └── main.py               # Evaluation script (to be added)
```

## Output Schema

The system generates `output.csv` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | User submitting the claim |
| image_paths | string | Semicolon-separated image paths |
| user_claim | string | Chat transcript about the issue |
| claim_object | string | car, laptop, or package |
| evidence_standard_met | boolean | Whether evidence is sufficient |
| evidence_standard_met_reason | string | Reason for evidence decision |
| risk_flags | string | Semicolon-separated risk flags |
| issue_type | string | Visible issue type |
| object_part | string | Relevant object part |
| claim_status | string | supported, contradicted, or not_enough_information |
| claim_status_justification | string | Concise explanation |
| supporting_image_ids | string | Image IDs supporting decision |
| valid_image | boolean | Whether images are usable |
| severity | string | none, low, medium, high, or unknown |

## Allowed Values

**claim_status**: supported, contradicted, not_enough_information

**issue_type**: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown

**risk_flags**: none, blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation, non_original_image, text_instruction_present, user_history_risk, manual_review_required

## Rate Limiting

The system implements:
- API key rotation across 3 keys
- Exponential backoff (10s, 20s, 40s, max 60s)
- 2-second delay between claims
- Automatic retry on 429 errors

## Error Handling

The system handles:
- Corrupted or unreadable image files
- API rate limits with retry logic
- JSON parsing errors from VLM responses
- Missing user history data
- Multi-image scenarios with partial failures

## Known Issues

- Some test images in `dataset/images/test/` are corrupted or in unsupported formats
- The system gracefully handles these by marking claims as "not_enough_information"
- Free tier Gemini API has rate limits (20 requests/minute for gemini-2.5-flash-lite)

## Evaluation

Run evaluation on sample claims:
```bash
python evaluation/main.py
```

This will generate metrics comparing predictions against ground truth in `dataset/sample_claims.csv`.

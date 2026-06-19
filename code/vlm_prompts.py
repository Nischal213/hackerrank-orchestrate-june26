"""
VLM Prompt Templates for Claim Verification
"""

# Main analysis prompt for single image
SINGLE_IMAGE_ANALYSIS_PROMPT = """You are a damage claim verification expert. Analyze the submitted image and the claim conversation to determine if the evidence supports the claim.

CLAIM OBJECT: {claim_object}
USER CLAIM CONVERSATION: {user_claim}

Analyze the image and provide the following information in JSON format:

{{
  "object_detected": "car|laptop|package|other",
  "object_part_visible": "specific part name or 'unknown'",
  "issue_type": "dent|scratch|crack|glass_shatter|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown",
  "damage_visible": true/false,
  "damage_description": "brief description of what is visible",
  "image_quality": {{
    "clear": true/false,
    "blurry": true/false,
    "low_light_or_glare": true/false,
    "cropped_or_obstructed": true/false,
    "wrong_angle": true/false
  }},
  "evidence_sufficient": true/false,
  "evidence_reason": "brief explanation",
  "supports_claim": true/false,
  "contradicts_claim": true/false,
  "severity": "none|low|medium|high|unknown"
}}

CRITICAL: Set supports_claim=true when:
1. The claimed object part is clearly visible
2. The claimed damage type is visible and matches the description
3. The image quality is sufficient to verify the claim
4. There is clear visual evidence supporting the user's claim

Set contradicts_claim=true when:
1. The visible damage clearly contradicts the claim description
2. The claimed part is visible but shows no damage when damage is claimed
3. The visible damage type is completely different from what is claimed

Set evidence_sufficient=true when:
1. The claimed part is visible enough to inspect
2. The image quality allows for damage assessment
3. There is enough visual information to make a determination

Set evidence_sufficient=false when:
1. The claimed part is not visible or is at wrong angle
2. Image is too blurry, dark, or obstructed to assess
3. There is insufficient visual information

IMPORTANT: Use these specific issue type rules:
- CRITICAL: Choose EXACTLY ONE string value from the allowed options for "issue_type" and "object_part_visible". Do NOT return multiple options joined by pipes (e.g., do NOT output "dent|scratch|broken_part"). If multiple distinct damage types are visible, select the single most prominent or severe one.
- Use "crack" for glass damage (windshield, laptop screen) showing crack lines - NOT "glass_shatter"
- Use "crack" for glass damage (windshield, laptop screen) showing crack lines - NOT "glass_shatter"
- Use "broken_part" when a component is physically broken, has missing material, is detached, or torn apart (headlight, mirror, hinge, bumper with holes, etc.)
- Use "dent" for surface deformation/indentation without material loss
- Use "scratch" for surface marks, paint damage, or scrapes without material loss or deformation
- Use "glass_shatter" only if glass is completely shattered into many pieces
- Use "missing_part" when an entire component is absent

Use these exact values for object parts:
- Car: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown
- Laptop: screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown
- Package: box, package_corner, package_side, seal, label, contents, item, unknown

Be precise and evidence-based. If the claimed part is not visible or the damage cannot be verified, state that clearly."""

# Multi-image analysis prompt
MULTI_IMAGE_ANALYSIS_PROMPT = """You are a damage claim verification expert. Analyze the submitted images and the claim conversation to determine if the evidence supports the claim.

CLAIM OBJECT: {claim_object}
USER CLAIM CONVERSATION: {user_claim}
NUMBER OF IMAGES: {num_images}

Analyze each image and provide the following information in JSON format:

{{
  "overall_assessment": {{
    "object_detected": "car|laptop|package|other",
    "claimed_part_visible_in_any_image": true/false,
    "damage_visible_in_any_image": true/false,
    "evidence_sufficient": true/false,
    "evidence_reason": "brief explanation",
    "supports_claim": true/false,
    "contradicts_claim": true/false
  }},
  "per_image_analysis": [
    {{
      "image_id": "img_1",
      "object_part_visible": "specific part name",
      "issue_type": "dent|scratch|crack|glass_shatter|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown",
      "damage_visible": true/false,
      "damage_description": "brief description",
      "image_quality": {{
        "clear": true/false,
        "blurry": true/false,
        "low_light_or_glare": true/false,
        "cropped_or_obstructed": true/false,
        "wrong_angle": true/false
      }},
      "supports_claim": true/false,
      "contradicts_claim": true/false
    }}
  ],
  "supporting_image_ids": ["img_1", "img_2"],
  "risk_flags": ["blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch", "possible_manipulation", "non_original_image", "text_instruction_present", "none"],
  "severity": "none|low|medium|high|unknown"
}}

IMPORTANT: Use these specific issue type rules:
- CRITICAL: Choose EXACTLY ONE string value from the allowed options for "issue_type" and "object_part_visible". Do NOT return multiple options joined by pipes (e.g., do NOT output "dent|scratch|broken_part"). If multiple distinct damage types are visible, select the single most prominent or severe one.
- Use "crack" for glass damage (windshield, laptop screen) showing crack lines - NOT "glass_shatter"
- Use "crack" for glass damage (windshield, laptop screen) showing crack lines - NOT "glass_shatter"
- Use "broken_part" when a component is physically broken, has missing material, is detached, or torn apart (headlight, mirror, hinge, bumper with holes, etc.)
- Use "dent" for surface deformation/indentation without material loss
- Use "scratch" for surface marks, paint damage, or scrapes without material loss or deformation
- Use "glass_shatter" only if glass is completely shattered into many pieces
- Use "missing_part" when an entire component is absent

Use these exact values for object parts:
- Car: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown
- Laptop: screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown
- Package: box, package_corner, package_side, seal, label, contents, item, unknown

Be precise and evidence-based. Consider all images together - at least one should clearly show the claimed part and damage."""

# Risk assessment prompt (to be used after image analysis)
RISK_ASSESSMENT_PROMPT = """Based on the image analysis and user history, assess any additional risks.

IMAGE ANALYSIS RESULT: {image_analysis}
USER HISTORY: {user_history}

Provide risk flags in JSON format:
{{
  "risk_flags": ["blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch", "possible_manipulation", "non_original_image", "text_instruction_present", "user_history_risk", "manual_review_required", "none"],
  "risk_reasoning": "brief explanation of why these flags apply"
}}

BE CONSERVATIVE with risk flags:
- Only add "manual_review_required" if there are clear signs of manipulation, severe image quality issues, or very concerning user history patterns
- Only add "claim_mismatch" if the visible damage clearly contradicts the claim description
- Only add "user_history_risk" if user history shows high rejection rate (>50%) or repeated fraudulent patterns
- Only add "possible_manipulation" if there are obvious signs of editing or non-original images
- If no significant risks exist, use ["none"]"""

# Evidence requirements check prompt
EVIDENCE_REQUIREMENTS_PROMPT = """Check if the image evidence meets the minimum requirements for this claim type.

CLAIM OBJECT: {claim_object}
ISSUE TYPE: {issue_type}
IMAGE ANALYSIS: {image_analysis}
EVIDENCE REQUIREMENT: {evidence_requirement}

Provide assessment in JSON format:
{{
  "evidence_standard_met": true/false,
  "evidence_standard_met_reason": "brief explanation referencing the requirement"
}}

Be strict - if the requirement specifies the part should be "visible clearly enough to inspect" and it's blurry or at wrong angle, mark as not met."""

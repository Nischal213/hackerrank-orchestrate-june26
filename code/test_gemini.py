import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image


def test_gemini_connection():
    """Test Gemini API connection and basic image analysis."""
    load_dotenv()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY_1")
    if not api_key or api_key == "your_first_api_key_here":
        print("ERROR: GEMINI_API_KEY_1 not configured in .env")
        sys.exit(1)

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Test with Gemini 2.5 Flash (faster, good for initial testing)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    print("Testing Gemini 2.5 Flash connection...")

    # Test with a simple text prompt first
    try:
        response = model.generate_content(
            "Hello, can you respond with 'Connection successful'?"
        )
        print(f"Text test response: {response.text}")
    except Exception as e:
        print(f"ERROR: Text generation failed: {e}")
        sys.exit(1)

    # Test image analysis with a sample image
    sample_image_path = "../dataset/images/sample/case_001/img_1.jpg"

    if not os.path.exists(sample_image_path):
        print(f"ERROR: Sample image not found at {sample_image_path}")
        sys.exit(1)

    try:
        img = Image.open(sample_image_path)
        print(f"Loaded sample image: {sample_image_path}")

        # Test basic image analysis
        prompt = """Analyze this image and describe:
1. What object is shown (car, laptop, package, or other)?
2. What part of the object is visible?
3. Is there any visible damage? If so, describe it.
4. What is the image quality (clear, blurry, good lighting, etc.)?

Keep your response concise."""

        response = model.generate_content([prompt, img])
        print(f"\nImage analysis response:\n{response.text}")

    except Exception as e:
        print(f"ERROR: Image analysis failed: {e}")
        sys.exit(1)

    print("\n✓ Gemini 2.5 Flash connection test successful!")


if __name__ == "__main__":
    test_gemini_connection()

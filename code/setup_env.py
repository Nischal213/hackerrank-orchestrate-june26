import os
import sys
from dotenv import load_dotenv


def setup_environment():
    """Setup Python environment and validate configuration."""
    load_dotenv()

    # Check if API keys are configured
    api_keys = [
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
    ]

    available_keys = [
        k
        for k in api_keys
        if k
        and k != "your_first_api_key_here"
        and k != "your_second_api_key_here"
        and k != "your_third_api_key_here"
    ]

    if not available_keys:
        print("ERROR: No valid Gemini API keys found in .env file")
        print("Please copy .env.example to .env and add your API keys")
        sys.exit(1)

    print(f"Found {len(available_keys)} valid API key(s)")

    # Test imports
    try:
        import google.generativeai as genai
        import pandas as pd
        from PIL import Image

        print("All required packages imported successfully")
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    return available_keys[0]  # Return first available key for testing


if __name__ == "__main__":
    setup_environment()

"""
Debug script to see what FinBERT actually returns
"""

from transformers import pipeline
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Test with various texts
test_texts = [
    "Apple stock is surging today",
    "Tesla faces major challenges",
    "The market is neutral",
    "SO",
    "NOW",
    "TECH",
    "Technology sector performs well",
    "Bank stocks decline",
]

print("Testing FinBERT sentiment analyzer:")
print("=" * 80)

for text in test_texts:
    try:
        result = sentiment_analyzer(text[:512])[0]
        print(f"\nText: '{text}'")
        print(f"  Result: {result}")
        print(f"  Label: {result.get('label')}")
        print(f"  Score: {result.get('score')}")
        print(f"  Label (upper): {str(result.get('label', '')).upper()}")
    except Exception as e:
        print(f"\nText: '{text}'")
        print(f"  Error: {e}")

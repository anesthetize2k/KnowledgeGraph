#!/usr/bin/env python3
"""Simple test for the extractor"""

from core.triplet_extractor import TripletExtractor

def test_simple():
    try:
        extractor = TripletExtractor()
        print("✅ Extractor initialized")
        
        # Simple test
        test_text = "Assassin's Creed Shadows is developed by Ubisoft Quebec"
        print(f"Testing: {test_text}")
        
        # Test the LLM call directly
        result = extractor._llm_extract(test_text)
        print(f"LLM result: {result}")
        
        extractor.close()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple()


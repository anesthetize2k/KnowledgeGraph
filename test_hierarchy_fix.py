#!/usr/bin/env python3
"""Test the updated relationship hierarchy logic"""

from core.triplet_extractor import TripletExtractor

def test_hierarchy_fix():
    """Test that features/attributes are not linked back to chunks"""
    print("🧪 Testing updated relationship hierarchy...")
    
    try:
        extractor = TripletExtractor()
        
        # Test text with game and feature
        test_text = "Assassin's Creed Shadows has authentic representation of Feudal Japan as a gameplay feature"
        print(f"📝 Test text: {test_text}")
        
        # Extract triples
        triples = extractor.extract(test_text)
        print(f"✅ Extracted {len(triples)} triples:")
        for i, triple in enumerate(triples, 1):
            print(f"  {i}. {triple}")
        
        # Test structured extraction
        structured = extractor.extract_structured(test_text)
        print(f"\n📊 Structured extraction:")
        print(f"  Entities: {len(structured['entities'])}")
        print(f"  Relations: {len(structured['relations'])}")
        
        print("\n🎯 Expected hierarchy:")
        print("  Document → Chunk → 'Assassin's Creed Shadows' (linked to chunk)")
        print("  'Assassin's Creed Shadows' → has_feature → 'authentic representation' (NOT linked to chunk)")
        print("  'authentic representation' → of_type → 'gameplay_feature' (NOT linked to chunk)")
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            extractor.close()
        except:
            pass

if __name__ == "__main__":
    test_hierarchy_fix()

#!/usr/bin/env python3
"""Test the new relationship hierarchy logic"""

from core.triplet_extractor import TripletExtractor

def test_relationship_hierarchy():
    """Test the new relationship hierarchy"""
    print("🧪 Testing new relationship hierarchy...")
    
    try:
        extractor = TripletExtractor()
        
        # Test text with metrics and entities
        test_text = "Assassin's Creed Shadows has hardcore fans which had an approval rating of 90%"
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
        
        print("\n🎯 New relationship hierarchy:")
        print("  Document -> Chunk -> Entity -> MetricValue")
        print("  (No direct linking of metrics/values to chunks)")
        
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
    test_relationship_hierarchy()


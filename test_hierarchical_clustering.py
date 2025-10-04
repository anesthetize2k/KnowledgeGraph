#!/usr/bin/env python3
"""Test the hierarchical clustering logic for primary entity identification"""

from core.triplet_extractor import TripletExtractor

def test_hierarchical_clustering():
    """Test hierarchical clustering logic"""
    print("🧪 Testing hierarchical clustering for primary entity identification...")
    
    try:
        extractor = TripletExtractor()
        
        # Test with a complex scenario
        test_text = """
        Assassin's Creed Shadows is developed by Ubisoft Quebec and published by Ubisoft. 
        The game features authentic representation of Feudal Japan with stealth mechanics, 
        parkour systems, and combat mechanics. It targets hardcore fans and casual players 
        in the North American market. The game has received positive reviews with 90% approval rating.
        """
        
        print(f"📝 Test text: {test_text.strip()}")
        
        # Extract structured data
        structured = extractor.extract_structured(test_text)
        entities = structured['entities']
        relations = structured['relations']
        
        print(f"\n📊 Extracted data:")
        print(f"  Entities: {len(entities)}")
        for i, entity in enumerate(entities, 1):
            print(f"    {i}. {entity['name']} ({entity['type']})")
        
        print(f"  Relations: {len(relations)}")
        for i, rel in enumerate(relations, 1):
            print(f"    {i}. {rel['head']} --{rel['relation']}--> {rel['tail']}")
        
        # Test hierarchical clustering
        primary_entities = extractor._identify_primary_entities(entities, relations)
        
        print(f"\n🎯 Hierarchical clustering results:")
        print(f"  Primary entities (linked to chunks): {len(primary_entities)}")
        for i, entity in enumerate(primary_entities, 1):
            print(f"    {i}. {entity}")
        
        print(f"\n✅ Expected behavior:")
        print(f"  - Most connected entities become primary")
        print(f"  - Their direct connections are excluded from primary")
        print(f"  - Process repeats until all entities are categorized")
        
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
    test_hierarchical_clustering()

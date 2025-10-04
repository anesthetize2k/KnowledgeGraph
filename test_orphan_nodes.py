#!/usr/bin/env python3
"""Test hierarchical clustering with orphaned nodes"""

from core.triplet_extractor import TripletExtractor

def test_orphan_nodes():
    """Test that orphaned nodes are properly handled"""
    print("🧪 Testing hierarchical clustering with orphaned nodes...")
    
    try:
        extractor = TripletExtractor()
        
        # Test case with orphaned nodes
        entities = [
            {"name": "Game A", "type": "installment"},
            {"name": "Studio B", "type": "studio"},
            {"name": "Feature C", "type": "content"},
            {"name": "Orphaned Entity", "type": "person"},  # This has no connections
            {"name": "Another Orphan", "type": "region"}    # This also has no connections
        ]
        
        relations = [
            {"head": "Game A", "tail": "Studio B"},
            {"head": "Game A", "tail": "Feature C"}
        ]
        
        print("📊 Test scenario:")
        print("  Entities: Game A, Studio B, Feature C, Orphaned Entity, Another Orphan")
        print("  Relations: Game A -> Studio B, Game A -> Feature C")
        print("  Expected: Game A (primary), Orphaned Entity (primary), Another Orphan (primary)")
        
        # Test the clustering logic
        primary_entities = extractor._identify_primary_entities(entities, relations)
        
        print(f"\n🎯 Results:")
        print(f"  Primary entities: {primary_entities}")
        print(f"  Count: {len(primary_entities)}")
        
        # Verify all entities are accounted for
        all_entity_names = {e["name"] for e in entities}
        primary_set = set(primary_entities)
        
        print(f"\n✅ Analysis:")
        print(f"  Total entities: {len(all_entity_names)}")
        print(f"  Primary entities: {len(primary_set)}")
        print(f"  All entities covered: {all_entity_names == primary_set}")
        
        # Check specific expectations
        expected_primary = {"Game A", "Orphaned Entity", "Another Orphan"}
        print(f"  Expected primary: {expected_primary}")
        print(f"  Actual primary: {primary_set}")
        print(f"  Matches expectation: {primary_set == expected_primary}")
        
        extractor.close()
        return primary_set == expected_primary
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_connected_orphans():
    """Test entities connected to already processed entities"""
    print("\n🧪 Testing entities connected to processed entities...")
    
    try:
        extractor = TripletExtractor()
        
        # Test case where some entities are connected to already processed ones
        entities = [
            {"name": "Game A", "type": "installment"},
            {"name": "Studio B", "type": "studio"},
            {"name": "Feature C", "type": "content"},
            {"name": "Late Entity", "type": "person"},
            {"name": "Another Late", "type": "region"}
        ]
        
        relations = [
            {"head": "Game A", "tail": "Studio B"},
            {"head": "Game A", "tail": "Feature C"},
            {"head": "Late Entity", "tail": "Another Late"}
        ]
        
        print("📊 Test scenario:")
        print("  Entities: Game A, Studio B, Feature C, Late Entity, Another Late")
        print("  Relations: Game A -> Studio B, Game A -> Feature C, Late Entity -> Another Late")
        print("  Expected: Game A (primary), Late Entity (primary), Another Late (primary)")
        
        primary_entities = extractor._identify_primary_entities(entities, relations)
        
        print(f"\n🎯 Results:")
        print(f"  Primary entities: {primary_entities}")
        
        # Should have Game A, Late Entity, and Another Late as primary
        expected_primary = {"Game A", "Late Entity", "Another Late"}
        actual_primary = set(primary_entities)
        
        print(f"  Expected: {expected_primary}")
        print(f"  Actual: {actual_primary}")
        print(f"  Correct: {actual_primary == expected_primary}")
        
        extractor.close()
        return actual_primary == expected_primary
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Orphaned Node Handling")
    print("=" * 50)
    
    test1_ok = test_orphan_nodes()
    test2_ok = test_connected_orphans()
    
    if test1_ok and test2_ok:
        print("\n🎉 All tests passed! Orphaned nodes are handled correctly.")
    else:
        print("\n⚠️ Some tests failed. Orphaned nodes may not be handled correctly.")

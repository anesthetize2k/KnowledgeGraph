#!/usr/bin/env python3
"""Test that no orphaned nodes are left after hierarchical clustering"""

from core.triplet_extractor import TripletExtractor

def test_no_orphans():
    """Test that all entities are properly linked"""
    print("🧪 Testing no orphaned nodes...")
    
    try:
        extractor = TripletExtractor()
        
        # Test case with various connection patterns
        entities = [
            {"name": "Game A", "type": "installment"},
            {"name": "Studio B", "type": "studio"},
            {"name": "Feature C", "type": "content"},
            {"name": "Orphan 1", "type": "person"},
            {"name": "Orphan 2", "type": "region"},
            {"name": "Connected Pair 1", "type": "platform"},
            {"name": "Connected Pair 2", "type": "platform"}
        ]
        
        relations = [
            {"head": "Game A", "tail": "Studio B"},
            {"head": "Game A", "tail": "Feature C"},
            {"head": "Connected Pair 1", "tail": "Connected Pair 2"}
        ]
        
        print("📊 Test scenario:")
        print("  Entities: Game A, Studio B, Feature C, Orphan 1, Orphan 2, Connected Pair 1, Connected Pair 2")
        print("  Relations: Game A -> Studio B, Game A -> Feature C, Connected Pair 1 -> Connected Pair 2")
        print("  Expected: ALL entities should be primary (no orphans allowed)")
        
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
        print(f"  No orphans: {len(all_entity_names - primary_set) == 0}")
        
        if all_entity_names == primary_set:
            print("🎉 SUCCESS: No orphaned nodes!")
            return True
        else:
            print(f"❌ FAILURE: Orphaned nodes found: {all_entity_names - primary_set}")
            return False
        
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
    print("🔧 Testing No Orphaned Nodes")
    print("=" * 50)
    
    success = test_no_orphans()
    
    if success:
        print("\n🎉 All tests passed! No orphaned nodes will be created.")
    else:
        print("\n⚠️ Tests failed. Orphaned nodes may still occur.")

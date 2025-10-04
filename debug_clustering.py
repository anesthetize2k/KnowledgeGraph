#!/usr/bin/env python3
"""Debug hierarchical clustering to see why entities aren't being linked to chunks"""

from core.triplet_extractor import TripletExtractor
from core.db_utils import ProcessedDocumentsDB

def debug_clustering():
    print("🔍 Debugging hierarchical clustering...")
    
    try:
        extractor = TripletExtractor()
        db = ProcessedDocumentsDB()
        
        # Get a sample chunk from the database
        with db._get_driver().session() as session:
            # Get a chunk with its entities
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.chunk_id IS NOT NULL
                RETURN c.chunk_id as chunk_id, c.text as text
                LIMIT 1
            """)
            
            chunk_record = result.single()
            if not chunk_record:
                print("❌ No chunks found in database")
                return
            
            chunk_id = chunk_record['chunk_id']
            text = chunk_record['text']
            
            print(f"📄 Testing with chunk: {chunk_id}")
            print(f"📝 Text preview: {text[:100]}...")
            
            # Test extraction
            result = extractor.extract_structured(text)
            entities = result["entities"]
            relations = result["relations"]
            
            print(f"\\n🧪 Extraction results:")
            print(f"  Entities: {len(entities)}")
            print(f"  Relations: {len(relations)}")
            
            if entities:
                print("\\n📋 Sample entities:")
                for i, entity in enumerate(entities[:5]):
                    print(f"  {i+1}. {entity['name']} ({entity['type']})")
                
                # Test hierarchical clustering
                print("\\n🔗 Testing hierarchical clustering...")
                primary_entities = extractor._identify_primary_entities(entities, relations)
                
                print(f"\\n🎯 Primary entities identified: {len(primary_entities)}")
                print(f"Primary entities: {primary_entities}")
                
                # Test linking
                if primary_entities:
                    print("\\n🔗 Testing entity linking...")
                    extractor._link_primary_entities_to_chunk(primary_entities, chunk_id, "test_doc")
                    
                    # Verify linking worked
                    verify_result = session.run("""
                        MATCH (e)-[:MENTIONED_IN_CHUNK]->(c:Chunk {chunk_id: $chunk_id})
                        RETURN count(e) as linked_count
                    """, chunk_id=chunk_id)
                    
                    linked_count = verify_result.single()['linked_count']
                    print(f"✅ Verification: {linked_count} entities linked to chunk")
                else:
                    print("❌ No primary entities identified!")
            else:
                print("❌ No entities extracted from text")
        
        extractor.close()
        db.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_clustering()
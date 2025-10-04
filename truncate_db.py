#!/usr/bin/env python3
"""Truncate the Neo4j database"""

from core.db_utils import ProcessedDocumentsDB

def truncate_database():
    """Truncate all data from Neo4j database"""
    print("🗑️ Truncating Neo4j database...")
    
    try:
        db = ProcessedDocumentsDB()
        
        with db._get_driver().session() as session:
            # Delete all nodes and relationships
            result = session.run('MATCH (n) DETACH DELETE n')
            print('✅ All nodes and relationships deleted')
            
            # Verify database is empty
            count_result = session.run('MATCH (n) RETURN count(n) as count')
            count = count_result.single()['count']
            print(f'📊 Remaining nodes: {count}')
            
            if count == 0:
                print('🎉 Database successfully truncated!')
            else:
                print('⚠️ Some nodes may still remain')
        
        db.close()
        print('🔗 Database connection closed')
        return True
        
    except Exception as e:
        print(f'❌ Error truncating database: {e}')
        return False

if __name__ == "__main__":
    truncate_database()

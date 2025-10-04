import os
from typing import Set, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class ProcessedDocumentsDB:
    """Manages processed documents in Neo4j database instead of local file."""
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
    
    def _get_driver(self):
        """Get Neo4j driver instance."""
        if self.driver is None:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self.driver
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def get_processed_files(self) -> Set[str]:
        """Get all processed file names from Neo4j."""
        try:
            with self._get_driver().session() as session:
                result = session.run(
                    """
                    MATCH (d:Document)
                    WHERE d.name IS NOT NULL
                    RETURN d.name as filename
                    """
                )
                processed_files = {record["filename"] for record in result}
                return processed_files
        except Exception as e:
            print(f"⚠️ Error retrieving processed files from Neo4j: {e}")
            return set()
    
    def is_file_processed(self, filename: str) -> bool:
        """Check if a specific file has been processed."""
        try:
            with self._get_driver().session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {name: $filename})
                    RETURN d.name as filename
                    """,
                    filename=filename
                )
                return result.single() is not None
        except Exception as e:
            print(f"⚠️ Error checking if file {filename} is processed: {e}")
            return False
    
    def mark_file_processed(self, filename: str, file_hash: str, file_path: str, 
                           file_size: int, content_preview: str) -> bool:
        """Mark a file as processed by creating/updating the Document node."""
        try:
            with self._get_driver().session() as session:
                session.run(
                    """
                    MERGE (d:Document {source_id: $source_id})
                    SET d.name = $name,
                        d.path = $path,
                        d.processed_at = datetime(),
                        d.file_size = $file_size,
                        d.content_preview = $content_preview
                    """,
                    source_id=file_hash,
                    name=filename,
                    path=file_path,
                    file_size=file_size,
                    content_preview=content_preview
                )
                return True
        except Exception as e:
            print(f"❌ Error marking file {filename} as processed: {e}")
            return False
    
    def get_unprocessed_files(self, all_files: list) -> list:
        """Get list of files that haven't been processed yet."""
        processed_files = self.get_processed_files()
        return [f for f in all_files if f not in processed_files]
    
    def get_processing_stats(self) -> dict:
        """Get statistics about processed documents."""
        try:
            with self._get_driver().session() as session:
                # Total processed documents
                total_result = session.run(
                    """
                    MATCH (d:Document)
                    RETURN count(d) as total_processed
                    """
                )
                total_processed = total_result.single()["total_processed"]
                
                # Documents processed today
                today_result = session.run(
                    """
                    MATCH (d:Document)
                    WHERE date(d.processed_at) = date()
                    RETURN count(d) as processed_today
                    """
                )
                processed_today = today_result.single()["processed_today"]
                
                # Total file size processed
                size_result = session.run(
                    """
                    MATCH (d:Document)
                    WHERE d.file_size IS NOT NULL
                    RETURN sum(d.file_size) as total_size
                    """
                )
                total_size = size_result.single()["total_size"] or 0
                
                return {
                    "total_processed": total_processed,
                    "processed_today": processed_today,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
        except Exception as e:
            print(f"⚠️ Error retrieving processing stats: {e}")
            return {
                "total_processed": 0,
                "processed_today": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0
            }
    
    def cleanup_orphaned_documents(self, valid_files: list) -> int:
        """Remove Document nodes for files that no longer exist in the filesystem."""
        try:
            with self._get_driver().session() as session:
                result = session.run(
                    """
                    MATCH (d:Document)
                    WHERE NOT d.name IN $valid_files
                    RETURN count(d) as orphaned_count
                    """,
                    valid_files=valid_files
                )
                orphaned_count = result.single()["orphaned_count"]
                
                if orphaned_count > 0:
                    session.run(
                        """
                        MATCH (d:Document)
                        WHERE NOT d.name IN $valid_files
                        DETACH DELETE d
                        """,
                        valid_files=valid_files
                    )
                    print(f"🧹 Cleaned up {orphaned_count} orphaned document nodes")
                
                return orphaned_count
        except Exception as e:
            print(f"⚠️ Error cleaning up orphaned documents: {e}")
            return 0

# Global instance for easy access
processed_docs_db = ProcessedDocumentsDB()

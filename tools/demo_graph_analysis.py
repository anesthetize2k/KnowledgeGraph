#!/usr/bin/env python3
"""
Demo script for the Graph Database Analyst tool.
Shows how to use the llm_analyze_graph_db.py tool.
"""

from llm_analyze_graph_db import GraphDBAnalyst

def demo_analysis():
    """Demonstrate the graph analysis tool."""
    print("🔍 Graph Database Analysis Demo")
    print("=" * 40)
    
    # Initialize the analyst
    analyst = GraphDBAnalyst()
    
    try:
        # Test database connection
        print("🔌 Testing database connection...")
        test_query = "MATCH (n) RETURN count(n) as total_nodes"
        results = analyst.execute_cypher_query(test_query)
        
        if "error" in results:
            print(f"❌ Database connection failed: {results['error']}")
            print("💡 Make sure your Neo4j database is running and credentials are correct in .env")
            return
        else:
            total_nodes = results['results'][0]['total_nodes']
            print(f"✅ Connected to database - {total_nodes} total nodes")
        
        # Run a simple analysis
        print("\n🔍 Running basic graph exploration...")
        
        # Query 1: Node types
        print("\n1. Analyzing node types...")
        node_types_query = "MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 10"
        node_results = analyst.execute_cypher_query(node_types_query)
        print(f"   Found {node_results['count']} different node type combinations")
        
        # Query 2: Relationship types
        print("\n2. Analyzing relationship types...")
        rel_types_query = "MATCH (n)-[r]->(m) RETURN type(r) as rel_types, count(r) as count ORDER BY count DESC LIMIT 10"
        rel_results = analyst.execute_cypher_query(rel_types_query)
        print(f"   Found {rel_results['count']} different relationship types")
        
        # Query 3: Sample data
        print("\n3. Sampling graph data...")
        sample_query = "MATCH (n) RETURN n LIMIT 5"
        sample_results = analyst.execute_cypher_query(sample_query)
        print(f"   Retrieved {sample_results['count']} sample nodes")
        
        print(f"\n✅ Basic analysis complete!")
        print(f"📄 Next analysis file will be: {analyst.analysis_file}")
        print("💡 To run full analysis with Claude Sonnet 4, use: python llm_analyze_graph_db.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
    finally:
        analyst.close()

if __name__ == "__main__":
    demo_analysis()

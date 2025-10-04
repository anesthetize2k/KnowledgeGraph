#!/usr/bin/env python3
"""Test the system after fixes"""

import sys
import os
sys.path.append(os.getcwd())

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing system imports...")
    
    try:
        # Test core imports
        from core.triplet_extractor import TripletExtractor
        print("✅ TripletExtractor import successful")
        
        from core.document_ingestor import DocumentIngestor
        print("✅ DocumentIngestor import successful")
        
        from core.main import process_new_files
        print("✅ Main processing import successful")
        
        # Test tools import
        from core.wiki_agent import WikiGenerationAgent
        print("✅ WikiGenerationAgent import successful")
        
        print("🎉 All imports working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triplet_extraction():
    """Test basic triplet extraction"""
    print("\n🧪 Testing triplet extraction...")
    
    try:
        from core.triplet_extractor import TripletExtractor
        extractor = TripletExtractor()
        
        # Simple test
        test_text = "Assassin's Creed Shadows is developed by Ubisoft Quebec"
        result = extractor.extract_structured(test_text)
        
        print(f"✅ Extraction successful: {len(result['entities'])} entities, {len(result['relations'])} relations")
        
        extractor.close()
        return True
        
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Knowledge Graph System")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test extraction
        extraction_ok = test_triplet_extraction()
        
        if extraction_ok:
            print("\n🎉 System is ready for processing!")
        else:
            print("\n⚠️ System has extraction issues")
    else:
        print("\n❌ System has import issues")

#!/usr/bin/env python3
"""Test processing a single file to debug hierarchical clustering"""

import os
from dotenv import load_dotenv
from core.main import _process_single_file

load_dotenv()

def test_single_file():
    print("🧪 Testing single file processing with debug output...")
    
    # Get the first file from data directory
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.endswith(('.docx', '.pdf', '.txt', '.md'))]
    
    if not files:
        print("❌ No files found in data directory")
        return
    
    test_file = files[0]
    print(f"📄 Testing with file: {test_file}")
    
    try:
        # Process the single file
        result = _process_single_file(test_file)
        print(f"✅ Processing result: {result}")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_file()

#!/usr/bin/env python3
"""
Knowledge Graph Processing System - Main Entry Point
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main function and other functions from core module
from core.main import main, process_new_files, query_graph, recreate_wiki_index

if __name__ == "__main__":
    main()
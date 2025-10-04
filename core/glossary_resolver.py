import json
import os
from typing import Dict, List, Tuple, Optional


class GlossaryResolver:
    """Utility for resolving synonyms and acronyms using the glossary.json file."""
    
    def __init__(self, glossary_path: str = "glossary.json"):
        self.glossary_path = glossary_path
        self.synonyms = self._load_glossary()
        self._build_reverse_mapping()
    
    def _load_glossary(self) -> Dict[str, List[str]]:
        """Load the glossary from JSON file."""
        if not os.path.exists(self.glossary_path):
            print(f"⚠️ Glossary file {self.glossary_path} not found. Creating empty glossary.")
            return {}
        
        try:
            with open(self.glossary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("synonyms", {})
        except Exception as e:
            print(f"❌ Error loading glossary: {e}")
            return {}
    
    def _build_reverse_mapping(self):
        """Build reverse mapping from synonym to canonical name."""
        self.synonym_to_canonical = {}
        for canonical, synonyms in self.synonyms.items():
            for synonym in synonyms:
                self.synonym_to_canonical[synonym.lower()] = canonical
    
    def resolve_query(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Resolve synonyms in a query and return the expanded query with mapping.
        
        Args:
            query: The original query string
            
        Returns:
            Tuple of (expanded_query, synonym_mapping)
        """
        expanded_query = query
        synonym_mapping = {}
        
        # Find and replace synonyms in the query
        for synonym, canonical in self.synonym_to_canonical.items():
            if synonym in query.lower():
                # Replace the synonym with canonical name
                expanded_query = expanded_query.replace(synonym, canonical)
                synonym_mapping[synonym] = canonical
        
        return expanded_query, synonym_mapping
    
    def get_canonical_name(self, name: str) -> str:
        """Get the canonical name for a given synonym."""
        return self.synonym_to_canonical.get(name.lower(), name)
    
    def get_synonyms(self, canonical_name: str) -> List[str]:
        """Get all synonyms for a canonical name."""
        return self.synonyms.get(canonical_name, [])
    
    def add_synonym(self, canonical_name: str, synonym: str):
        """Add a new synonym to the glossary."""
        if canonical_name not in self.synonyms:
            self.synonyms[canonical_name] = []
        
        if synonym not in self.synonyms[canonical_name]:
            self.synonyms[canonical_name].append(synonym)
            self.synonym_to_canonical[synonym.lower()] = canonical_name
            self._save_glossary()
    
    def _save_glossary(self):
        """Save the updated glossary to file."""
        data = {
            "synonyms": self.synonyms,
            "description": "Reference glossary for common synonyms and acronyms. This is for reference only - actual synonym relationships should be extracted from text using the 'is_synonym_of' relationship type."
        }
        
        try:
            with open(self.glossary_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving glossary: {e}")
    
    def expand_entity_names(self, entity_names: List[str]) -> List[str]:
        """Expand a list of entity names to include their synonyms."""
        expanded = set()
        for name in entity_names:
            expanded.add(name)  # Add original name
            # Add all synonyms for this canonical name
            synonyms = self.get_synonyms(name)
            expanded.update(synonyms)
        return list(expanded)

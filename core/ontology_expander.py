import json
import os
from typing import Dict, List, Set, Tuple, Optional
from .litellm_wrapper import LiteLLMChat


class OntologyExpander:
    """Analyzes extracted triples and suggests new ontology types using LLM."""
    
    def __init__(self, ontology_path: str = "ontology.json", suggestions_path: str = "ontology_suggestions.json"):
        self.ontology_path = ontology_path
        self.suggestions_path = suggestions_path
        self.llm = LiteLLMChat(model="gpt-4o-mini")  # Explicitly use mini for ontology suggestions
        self.current_ontology = self._load_ontology()
        self.suggestions = self._load_suggestions()
    
    def _load_ontology(self) -> Dict:
        """Load current ontology."""
        if not os.path.exists(self.ontology_path):
            return {"entity_types": [], "relation_types": []}
        
        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading ontology: {e}")
            return {"entity_types": [], "relation_types": []}
    
    def _load_suggestions(self) -> Dict:
        """Load existing suggestions."""
        if not os.path.exists(self.suggestions_path):
            return {"pending": [], "approved": [], "rejected": []}
        
        try:
            with open(self.suggestions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading suggestions: {e}")
            return {"pending": [], "approved": [], "rejected": []}
    
    def _save_suggestions(self):
        """Save suggestions to file."""
        try:
            with open(self.suggestions_path, 'w', encoding='utf-8') as f:
                json.dump(self.suggestions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving suggestions: {e}")
    
    def analyze_triple_gaps(self, triples: List[Dict]) -> List[Dict]:
        """
        Analyze triples that don't fit current ontology and suggest new types.
        
        Args:
            triples: List of triple dicts with keys: head, head_type, relation, tail, tail_type
            
        Returns:
            List of suggestion dicts
        """
        current_entity_types = set(self.current_ontology.get("entity_types", []))
        current_relation_types = set(self.current_ontology.get("relation_types", []))
        
        # Find triples with unknown types
        unknown_entities = set()
        unknown_relations = set()
        
        for triple in triples:
            head_type = triple.get("head_type", "")
            tail_type = triple.get("tail_type", "")
            relation = triple.get("relation", "")
            
            if head_type and head_type not in current_entity_types:
                unknown_entities.add(head_type)
            if tail_type and tail_type not in current_entity_types:
                unknown_entities.add(tail_type)
            if relation and relation not in current_relation_types:
                unknown_relations.add(relation)
        
        suggestions = []
        
        # Generate suggestions for unknown entity types
        if unknown_entities:
            entity_suggestions = self._suggest_entity_types(list(unknown_entities), triples)
            suggestions.extend(entity_suggestions)
        
        # Generate suggestions for unknown relation types
        if unknown_relations:
            relation_suggestions = self._suggest_relation_types(list(unknown_relations), triples)
            suggestions.extend(relation_suggestions)
        
        return suggestions
    
    def _suggest_entity_types(self, unknown_types: List[str], triples: List[Dict]) -> List[Dict]:
        """Use LLM to suggest new entity types."""
        if not unknown_types:
            return []
        
        # Create context from triples using these types
        context_triples = []
        for triple in triples:
            if (triple.get("head_type") in unknown_types or 
                triple.get("tail_type") in unknown_types):
                context_triples.append(f"{triple.get('head')} ({triple.get('head_type')}) --[{triple.get('relation')}]--> {triple.get('tail')} ({triple.get('tail_type')})")
        
        context = "\n".join(context_triples[:20])  # Limit context size
        
        prompt = f"""
You are an ontology expert. Analyze the following triples that contain unknown entity types and suggest appropriate new entity types for the ontology.

Unknown entity types: {', '.join(unknown_types)}

Current ontology entity types: {', '.join(self.current_ontology.get("entity_types", []))}

Example triples with unknown types:
{context}

For each unknown entity type, suggest:
1. Whether it should be added to the ontology (high confidence only)
2. A canonical name for the entity type
3. Brief description of what this entity type represents
4. Confidence level (0.0-1.0)

Answer ONLY with JSON in this exact format:
{{
  "suggestions": [
    {{
      "original_type": "string",
      "canonical_name": "string", 
      "description": "string",
      "confidence": 0.0-1.0,
      "should_add": true/false,
      "reasoning": "string"
    }}
  ]
}}

IMPORTANT: Only suggest adding types with confidence >= 0.8 and clear domain relevance.
"""
        
        try:
            response = self.llm.invoke(prompt)
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            suggestions = []
            
            for suggestion in data.get("suggestions", []):
                if suggestion.get("should_add", False) and suggestion.get("confidence", 0) >= 0.8:
                    suggestions.append({
                        "type": "entity",
                        "original_type": suggestion["original_type"],
                        "canonical_name": suggestion["canonical_name"],
                        "description": suggestion["description"],
                        "confidence": suggestion["confidence"],
                        "reasoning": suggestion["reasoning"],
                        "status": "pending"
                    })
            
            return suggestions
            
        except Exception as e:
            print(f"❌ Error generating entity type suggestions: {e}")
            return []
    
    def _suggest_relation_types(self, unknown_types: List[str], triples: List[Dict]) -> List[Dict]:
        """Use LLM to suggest new relation types."""
        if not unknown_types:
            return []
        
        # Create context from triples using these relation types
        context_triples = []
        for triple in triples:
            if triple.get("relation") in unknown_types:
                context_triples.append(f"{triple.get('head')} ({triple.get('head_type')}) --[{triple.get('relation')}]--> {triple.get('tail')} ({triple.get('tail_type')})")
        
        context = "\n".join(context_triples[:20])  # Limit context size
        
        prompt = f"""
You are an ontology expert. Analyze the following triples that contain unknown relation types and suggest appropriate new relation types for the ontology.

Unknown relation types: {', '.join(unknown_types)}

Current ontology relation types: {', '.join(self.current_ontology.get("relation_types", []))}

Example triples with unknown relations:
{context}

For each unknown relation type, suggest:
1. Whether it should be added to the ontology (high confidence only)
2. A canonical name for the relation type
3. Brief description of what this relation represents
4. Confidence level (0.0-1.0)

Answer ONLY with JSON in this exact format:
{{
  "suggestions": [
    {{
      "original_type": "string",
      "canonical_name": "string", 
      "description": "string",
      "confidence": 0.0-1.0,
      "should_add": true/false,
      "reasoning": "string"
    }}
  ]
}}

IMPORTANT: Only suggest adding relations with confidence >= 0.8 and clear semantic meaning.
"""
        
        try:
            response = self.llm.invoke(prompt)
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            suggestions = []
            
            for suggestion in data.get("suggestions", []):
                if suggestion.get("should_add", False) and suggestion.get("confidence", 0) >= 0.8:
                    suggestions.append({
                        "type": "relation",
                        "original_type": suggestion["original_type"],
                        "canonical_name": suggestion["canonical_name"],
                        "description": suggestion["description"],
                        "confidence": suggestion["confidence"],
                        "reasoning": suggestion["reasoning"],
                        "status": "pending"
                    })
            
            return suggestions
            
        except Exception as e:
            print(f"❌ Error generating relation type suggestions: {e}")
            return []
    
    def add_suggestion(self, suggestion: Dict):
        """Add a new suggestion to the pending list."""
        suggestion["id"] = len(self.suggestions["pending"]) + len(self.suggestions["approved"]) + len(self.suggestions["rejected"]) + 1
        suggestion["status"] = "pending"
        self.suggestions["pending"].append(suggestion)
        self._save_suggestions()
    
    def approve_suggestion(self, suggestion_id: int) -> bool:
        """Approve a suggestion and add it to the ontology."""
        for i, suggestion in enumerate(self.suggestions["pending"]):
            if suggestion["id"] == suggestion_id:
                # Move to approved
                approved_suggestion = self.suggestions["pending"].pop(i)
                approved_suggestion["status"] = "approved"
                self.suggestions["approved"].append(approved_suggestion)
                
                # Add to ontology
                if approved_suggestion["type"] == "entity":
                    if approved_suggestion["canonical_name"] not in self.current_ontology["entity_types"]:
                        self.current_ontology["entity_types"].append(approved_suggestion["canonical_name"])
                elif approved_suggestion["type"] == "relation":
                    if approved_suggestion["canonical_name"] not in self.current_ontology["relation_types"]:
                        self.current_ontology["relation_types"].append(approved_suggestion["canonical_name"])
                
                # Save both files
                self._save_ontology()
                self._save_suggestions()
                return True
        
        return False
    
    def reject_suggestion(self, suggestion_id: int) -> bool:
        """Reject a suggestion."""
        for i, suggestion in enumerate(self.suggestions["pending"]):
            if suggestion["id"] == suggestion_id:
                rejected_suggestion = self.suggestions["pending"].pop(i)
                rejected_suggestion["status"] = "rejected"
                self.suggestions["rejected"].append(rejected_suggestion)
                self._save_suggestions()
                return True
        
        return False
    
    def _save_ontology(self):
        """Save updated ontology to file."""
        try:
            with open(self.ontology_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_ontology, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving ontology: {e}")
    
    def get_pending_suggestions(self) -> List[Dict]:
        """Get all pending suggestions."""
        return self.suggestions["pending"]
    
    def get_suggestion_summary(self) -> Dict:
        """Get summary of all suggestions."""
        return {
            "pending": len(self.suggestions["pending"]),
            "approved": len(self.suggestions["approved"]),
            "rejected": len(self.suggestions["rejected"])
        }

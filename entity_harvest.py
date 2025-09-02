# entity_harvest.py
import json
import re
from typing import List, Dict
from litellm_wrapper import LiteLLMChat

ALLOWED_ENTITY_TYPES = [
    "brand","installment","studio","company","publisher","platform","market","person"
]

PROMPT = """# ROLE: Entity Extraction Specialist

You are a meticulous research assistant tasked with extracting only explicitly mentioned entities from a document passage. Your job is to identify proper nouns and specific entities that are directly named in the text.

## TASK: Extract entities from passage

## CHAIN OF THOUGHT PROCESS:

### Step 1: Read and Analyze
Carefully read through the passage below and identify:
- What proper nouns are explicitly mentioned by name
- What specific brands, companies, people, or platforms are directly referenced
- What entities are NOT mentioned (important for avoiding inference)

### Step 2: Entity Classification
For each explicitly mentioned entity, determine:
- What is the canonical name (exact spelling as it appears)
- What type category does it belong to
- Is it directly named in the passage (not just implied or referenced)

### Step 3: Validation
Before including any entity, verify:
- Is it explicitly mentioned by name in the passage?
- Can you point to the exact location where it appears?
- Is it a proper noun or specific entity (not a generic term)?

## STRICT EXTRACTION RULES:

### ZERO TOLERANCE FOR INFERENCE:
- **ONLY** extract entities that are EXPLICITLY mentioned by name in the passage
- **NEVER** infer or assume entities that might be related but aren't directly named
- **NEVER** extract generic terms like "brand", "company", "studio" without specific names
- **NEVER** extract entities that are only implied or referenced indirectly

### ENTITY TYPES:
- type ∈ {{brand, installment, studio, company, publisher, platform, market, person}}
- Use canonical names exactly as they appear (e.g., "Assassin's Creed", "PlayStation 5")
- Do NOT include vague tokens like "brand", "these elements", "intro"

### VALIDATION CHECKLIST:
- [ ] Entity is explicitly mentioned by name in the passage
- [ ] Entity is a proper noun or specific brand/company name
- [ ] Entity can be classified into one of the allowed types
- [ ] Entity name is spelled exactly as it appears in the text

## OUTPUT FORMAT:
Return ONLY JSON with this exact structure:
{{
  "entities": [
    {{"name":"Exact Name as Appears","type":"entity_type"}},
    {{"name":"Another Entity Name","type":"entity_type"}}
  ]
}}

## SOURCE PASSAGE:
\"\"\"{text}\"\"\"

## FINAL CHECK:
Before submitting, verify that every entity in your response is explicitly mentioned by name in the passage above. If you cannot point to the exact location of an entity name, do not include it.
"""

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip(" '\"\"®")).strip()

class EntityHarvest:
    def __init__(self):
        self.llm = LiteLLMChat()

    def harvest(self, text: str, max_chars=3000) -> List[Dict]:
        print(f"      🤖 Calling LLM for entity extraction...")
        print(f"      📝 Text preview: {repr(text[:100])}...")
        
        try:
            raw = self.llm.invoke(PROMPT.format(text=text[:max_chars]))
            print(f"      📄 Raw LLM response: {repr(raw[:200])}...")
        except Exception as e:
            print(f"      ❌ LLM call failed: {e}")
            print(f"      ❌ Error type: {type(e)}")
            print(f"      ❌ Error repr: {repr(e)}")
            print(f"      ❌ Full error details: {e.__class__.__name__}: {str(e)}")
            import traceback
            print(f"      ❌ Traceback: {traceback.format_exc()}")
            return []
        
        cleaned = raw.strip()

        # Remove markdown fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip("` \n")

        # Extract JSON object strictly
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0).strip()

        try:
            data = json.loads(cleaned)
            ents = data.get("entities", [])
            print(f"      📊 Parsed {len(ents)} entities from JSON")
        except Exception as e:
            print(f"      ⚠️ EntityHarvest parse failed: {e}")
            print(f"      📄 Raw output was: {raw[:300]}...")
            print(f"      🔧 Cleaned output was: {repr(cleaned[:300])}...")
            # Safe fallback
            return []

        out = []
        for e in ents:
            name = _clean(e.get("name",""))
            et   = (e.get("type","") or "").lower().strip()
            if not name or et not in ALLOWED_ENTITY_TYPES:
                print(f"      ⚠️ Skipping invalid entity: name='{name}', type='{et}'")
                continue
            out.append({"name": name, "type": et})
            print(f"      ✅ Valid entity: {name} ({et})")
        
        print(f"      🎯 Final entity count: {len(out)}")
        return out

# entity_harvest.py
import json
import re
from typing import List, Dict
from litellm_wrapper import LiteLLMChat

ALLOWED_ENTITY_TYPES = [
    "brand","installment","studio","company","publisher","platform","market","person"
]

PROMPT = """Extract canonical entities mentioned in the passage.

Return ONLY JSON like:
{
  "entities": [
    {"name":"Assassin's Creed","type":"brand"},
    {"name":"Ubisoft","type":"company"},
    {"name":"Ubisoft Quebec","type":"studio"},
    {"name":"Yves Guillemot","type":"person"},
    {"name":"PlayStation 5","type":"platform"}
  ]
}

Rules:
- type ∈ {brand, installment, studio, company, publisher, platform, market, person}
- Use canonical names (e.g., "Assassin's Creed", "PlayStation 5").
- Do NOT include vague tokens like "brand", "these elements", "intro".
- If unsure, omit it.
Passage:
\"\"\"{text}\"\"\"
"""

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip(" '\"“"™®")).strip()

class EntityHarvest:
    def __init__(self):
        self.llm = LiteLLMChat()

    def harvest(self, text: str, max_chars=3000) -> List[Dict]:
        print(f"  → Calling LLM with prompt...")
        try:
            raw = self.llm.invoke(PROMPT.format(text=text[:max_chars]))
            print(f"  → Raw LLM response: {repr(raw)}")
        except Exception as e:
            print(f"  ❌ LLM call failed: {e}")
            print(f"  ❌ Error type: {type(e)}")
            print(f"  ❌ Error repr: {repr(e)}")
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
        except Exception as e:
            print("⚠️ EntityHarvest parse failed:", e)
            print("Raw output was:\n", raw)
            print("Cleaned output was:\n", repr(cleaned))
            # Safe fallback
            return []

        out = []
        for e in ents:
            name = _clean(e.get("name",""))
            et   = (e.get("type","") or "").lower().strip()
            if not name or et not in ALLOWED_ENTITY_TYPES:
                continue
            out.append({"name": name, "type": et})
        return out

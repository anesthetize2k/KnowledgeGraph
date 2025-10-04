# wiki_creator.py
import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from .litellm_wrapper import LiteLLMChat

load_dotenv()
from neo4j import GraphDatabase

def safe_json_loads(json_str: str, fallback=None):
    """
    Safely parse JSON with comprehensive "Extra data" error handling and format validation.
    """
    if not json_str or not isinstance(json_str, str):
        print(f"🔍 safe_json_loads: Invalid input - not string or empty")
        return fallback
    
    # Clean up common issues
    cleaned = json_str.strip()
    original_length = len(cleaned)
    
    # Remove markdown code blocks if present
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
        print(f"🔍 safe_json_loads: Removed ```json prefix")
    if cleaned.startswith('```'):
        cleaned = cleaned[3:]
        print(f"🔍 safe_json_loads: Removed ``` prefix")
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
        print(f"🔍 safe_json_loads: Removed ``` suffix")
    cleaned = cleaned.strip()
    
    # Check for empty string after cleaning
    if not cleaned:
        print(f"🔍 safe_json_loads: Empty string after cleaning")
        return fallback
    
    print(f"🔍 safe_json_loads: Original length: {original_length}, Cleaned length: {len(cleaned)}")
    
    # PROACTIVE "Extra data" check: Find the end of the first complete JSON object
    # and extract only that part to avoid "Extra data" errors
    try:
        brace_count = 0
        json_end = 0
        for i, char in enumerate(cleaned):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        if json_end > 0 and json_end < len(cleaned):
            # There's extra data after the JSON object, extract only the JSON part
            extra_data = cleaned[json_end:]
            cleaned = cleaned[:json_end]
            print(f"🔍 safe_json_loads: Found extra data after JSON object (length: {len(extra_data)})")
            print(f"🔍 safe_json_loads: Extra data preview: {extra_data[:100]}...")
        elif json_end == 0:
            # No complete JSON object found, this might be malformed
            print(f"🔍 safe_json_loads: No complete JSON object found, string might be malformed")
    except Exception as e:
        print(f"🔍 safe_json_loads: Error in proactive check: {e}")
    
    try:
        result = json.loads(cleaned)
        print(f"🔍 safe_json_loads: Successfully parsed JSON")
        return result
    except json.JSONDecodeError as e:
        error_msg = str(e)
        print(f"🔍 safe_json_loads: JSONDecodeError: {error_msg}")
        print(f"🔍 safe_json_loads: Cleaned string length: {len(cleaned)}")
        print(f"🔍 safe_json_loads: Cleaned string preview: {cleaned[:200]}...")
        
        # Handle "Extra data" errors with more aggressive extraction
        if "Extra data" in error_msg:
            print(f"🔍 safe_json_loads: Extra data detected, trying aggressive extraction...")
            try:
                # Try to find the first complete JSON object by looking for the pattern
                import re
                # Look for the first complete JSON object using regex
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    json_part = json_match.group(0)
                    print(f"🔍 safe_json_loads: Found JSON object via regex, length: {len(json_part)}")
                    result = json.loads(json_part)
                    print(f"🔍 safe_json_loads: Successfully parsed extracted JSON")
                    return result
            except Exception as regex_error:
                print(f"🔍 safe_json_loads: Regex extraction failed: {regex_error}")
        
        # Handle "Expecting value" errors (empty or invalid JSON)
        if "Expecting value" in error_msg:
            print(f"🔍 safe_json_loads: Trying to find valid JSON structure...")
            # Try to find any valid JSON structure
            for start in range(min(len(cleaned), 100)):  # Limit search to first 100 chars
                if cleaned[start] in '{[':
                    try:
                        result = json.loads(cleaned[start:])
                        print(f"🔍 safe_json_loads: Found valid JSON at position {start}")
                        return result
                    except:
                        continue
        
        # Handle "unterminated string" errors
        if "unterminated string" in error_msg:
            print(f"🔍 safe_json_loads: Trying to fix unterminated string...")
            # Try to fix common string issues
            try:
                # Remove trailing commas and fix quotes
                fixed = re.sub(r',\s*}', '}', cleaned)
                fixed = re.sub(r',\s*]', ']', fixed)
                result = json.loads(fixed)
                print(f"🔍 safe_json_loads: Fixed unterminated string")
                return result
            except Exception as fix_error:
                print(f"🔍 safe_json_loads: Failed to fix unterminated string: {fix_error}")
        
        print(f"🔍 safe_json_loads: All attempts failed, returning fallback")
        return fallback
    except Exception as e:
        print(f"🔍 safe_json_loads: Unexpected error: {e}")
        return fallback

def canonical_name_from_graph(name: str, label: str, session) -> str:
    """
    Checks if an entity already exists in Neo4j with the given label and name.
    If found, returns the canonical stored name (e.g., with proper casing).
    Otherwise returns the input.
    """
    rec = session.run(
        f"MATCH (n:{label} {{name:$name}}) RETURN n.name AS name",
        name=name
    ).single()
    return rec["name"] if rec else name

WIKI_FOLDER = "Wikis"
WIKI_SCHEMA_PATH = "wiki_schema.json"
SCHEMA_SUGGESTIONS_PATH = "schema_update_suggestions.json"

DEFAULT_CLASSIFY_TOPK = 3
CONFIDENCE_THRESHOLD = 0.60  # matches below this are ignored

# ===== gating & candidates config =====
CANDIDATES_PATH = "wiki_candidates.json"
MIN_FACTS_FOR_WIKI = int(os.getenv("MIN_FACTS_FOR_WIKI", "2"))
REQUIRE_UBISOFT_PERTINENCE = True
CREATE_EMPTY_STUBS = False  # set True if you want empty stubs for non-merit items


def _safe_slug(s: str) -> str:
    s = s.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-]", "", s)
    return s


class WikiCreator:
    """
    - Classifies a document against existing wiki schema using LLM.
    - If no good match, proposes NEW_WIKI_TYPE and records to schema_update_suggestions.json (approved: pending).
    - If match(es), generates Markdown for each matching wiki using LLM and saves under Wikis/.
    - Adds YAML front matter with metadata including approved status, subtype, source file, etc.
    """

    def __init__(self):
        os.makedirs(WIKI_FOLDER, exist_ok=True)
        self.llm = LiteLLMChat()
        self.schema = self._load_schema()
        self.suggestions = self._load_suggestions()

    def _load_candidates(self) -> Dict:
        if os.path.exists(CANDIDATES_PATH):
            with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                return safe_json_loads(content, {"candidates": []})
        return {"candidates": []}

    def _save_candidates(self, data: Dict):
        with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_wiki_candidate(self, entry: Dict):
        """
        entry keys: entity_name, wiki_type, subtype, file, facts_count, ubisoft_linked, merit, reason
        Dedup by (entity_name, wiki_type, file).
        """
        data = self._load_candidates()
        key = (entry.get("entity_name"), entry.get("wiki_type"), entry.get("file"))
        seen = {(c.get("entity_name"), c.get("wiki_type"), c.get("file")) for c in data["candidates"]}
        if key not in seen:
            data["candidates"].append(entry)
            self._save_candidates(data)

    def extract_entity_facts(self, entity_name: str, doc_text: str) -> Dict:
        """
        Ask LLM to pull only explicit facts about `entity_name` from doc_text.
        Returns dict: { "facts": [str...], "mentions": int }
        """
        print(f"🔍 extract_entity_facts: Extracting facts for {entity_name}")
        prompt = f"""
You will extract explicit facts about a target entity from the given text.
Target entity: "{entity_name}"

Rules:
- Extract only facts explicitly stated in the text (no world knowledge, no inferences).
- Return short atomic bullets. If none exist, return an empty list.
- Also count approximate mentions of the entity name (0..N).
- Answer ONLY JSON: {{ "facts": [..], "mentions": <int> }}

Text:
\"\"\"{doc_text[:4000]}\"\"\""""
        print(f"🔍 extract_entity_facts: Calling LLM for {entity_name}")
        raw = self.llm.invoke(prompt)
        print(f"🔍 extract_entity_facts: LLM response length: {len(raw)}")
        try:
            # Clean up the response - remove markdown code blocks if present
            cleaned = raw.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = safe_json_loads(cleaned, {"facts": [], "mentions": 0})
            facts = data.get("facts", [])
            mentions = int(data.get("mentions", 0))
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                # Try to extract just the JSON part before the extra data
                try:
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(cleaned):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end > 0:
                        json_part = cleaned[:json_end]
                        data = safe_json_loads(json_part, {"facts": [], "mentions": 0})
                        facts = data.get("facts", [])
                        mentions = int(data.get("mentions", 0))
                    else:
                        facts, mentions = [], 0
                except Exception:
                    facts, mentions = [], 0
            else:
                print(f"    ⚠️ JSON parsing failed for {entity_name}: {e}")
                print(f"    Raw response: {raw[:200]}...")
                facts, mentions = [], 0
        except Exception as e:
            print(f"    ⚠️ JSON parsing failed for {entity_name}: {e}")
            print(f"    Raw response: {raw[:200]}...")
            facts, mentions = [], 0
        return {"facts": facts, "mentions": mentions}

    def judge_ubisoft_pertinence(self, entity_name: str, etype: str, doc_text: str) -> Dict:
        """
        Decide if the document ties the entity to Ubisoft context:
        - Ubisoft as company/studio/publisher
        - Ubisoft brands (e.g., Assassin's Creed) or Ubisoft projects
        Return: { "ubisoft_linked": bool, "why": str }
        """
        print(f"🔍 judge_ubisoft_pertinence: Judging {entity_name} ({etype})")
        prompt = f"""
Decide if the following text connects the entity to Ubisoft's ecosystem.

Entity: "{entity_name}" (type: {etype})
Question: Is there an explicit connection in the text to Ubisoft (company/studios/publishing)
or to a Ubisoft brand/franchise (e.g., Assassin's Creed), or to Ubisoft projects?

Return ONLY JSON: {{ "ubisoft_linked": true|false, "why": "<short reason or empty>" }}

Text:
\"\"\"{doc_text[:3000]}\"\"\""""
        print(f"🔍 judge_ubisoft_pertinence: Calling LLM for {entity_name}")
        raw = self.llm.invoke(prompt)
        print(f"🔍 judge_ubisoft_pertinence: LLM response length: {len(raw)}")
        try:
            # Clean up the response - remove markdown code blocks if present
            cleaned = raw.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = safe_json_loads(cleaned, {"ubisoft_linked": False, "why": ""})
            return {
                "ubisoft_linked": bool(data.get("ubisoft_linked", False)),
                "why": str(data.get("why", "")).strip()
            }
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                # Try to extract just the JSON part before the extra data
                try:
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(cleaned):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end > 0:
                        json_part = cleaned[:json_end]
                        data = safe_json_loads(json_part, {"ubisoft_linked": False, "why": ""})
                        return {
                            "ubisoft_linked": bool(data.get("ubisoft_linked", False)),
                            "why": str(data.get("why", "")).strip()
                        }
                    else:
                        return {"ubisoft_linked": False, "why": ""}
                except Exception:
                    return {"ubisoft_linked": False, "why": ""}
            else:
                print(f"    ⚠️ JSON parsing failed for ubisoft pertinence: {e}")
                print(f"    Raw response: {raw[:200]}...")
                return {"ubisoft_linked": False, "why": ""}
        except Exception as e:
            print(f"    ⚠️ JSON parsing failed for ubisoft pertinence: {e}")
            print(f"    Raw response: {raw[:200]}...")
            return {"ubisoft_linked": False, "why": ""}

    def assess_entity_worthiness(self, entity: dict, doc_text: str, source_file: str) -> Dict:
        """
        Consolidate facts + pertinence to judge if this entity merits a wiki now.
        """
        name = entity["name"]
        etype = entity.get("type", "brand")
        facts_info = self.extract_entity_facts(name, doc_text)
        pertinence = self.judge_ubisoft_pertinence(name, etype, doc_text)

        facts_count = len(facts_info["facts"])
        ubisoft_linked = pertinence["ubisoft_linked"]
        merit = (facts_count >= MIN_FACTS_FOR_WIKI) and (ubisoft_linked or not REQUIRE_UBISOFT_PERTINENCE)
        


        reason = []
        reason.append(f"{facts_count} explicit fact(s)")
        reason.append("linked to Ubisoft" if ubisoft_linked else "no Ubisoft linkage")
        decision = {
            "entity_name": name,
            "wiki_type": self._map_entity_to_wiki_type(etype),
            "subtype": None,
            "file": source_file,
            "facts_count": facts_count,
            "ubisoft_linked": ubisoft_linked,
            "merit": merit,
            "reason": "; ".join(reason)
        }
        # always record to candidates list for transparency
        self.record_wiki_candidate(decision)
        return decision

    # ---------- schema & suggestions ----------
    def _load_schema(self) -> Dict:
        if not os.path.exists(WIKI_SCHEMA_PATH):
            raise FileNotFoundError(f"{WIKI_SCHEMA_PATH} not found. Please add it.")
        with open(WIKI_SCHEMA_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            data = safe_json_loads(content, {"wiki_types": {}})
        return data.get("wiki_types", {})

    def _save_schema(self):
        with open(WIKI_SCHEMA_PATH, "w", encoding="utf-8") as f:
            json.dump({"wiki_types": self.schema}, f, indent=2, ensure_ascii=False)

    def _load_suggestions(self) -> Dict:
        if os.path.exists(SCHEMA_SUGGESTIONS_PATH):
            with open(SCHEMA_SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                return safe_json_loads(content, {"suggestions": []})
        return {"suggestions": []}

    def _save_suggestions(self):
        with open(SCHEMA_SUGGESTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.suggestions, f, indent=2, ensure_ascii=False)

# --- inside wiki_creator.py, add this method in the WikiCreator class ---

    def propose_new_type(self, doc_text: str, file_name: str) -> dict:
        """
        If classify_document() returns no good matches, call this to ask the LLM
        for a NEW_WIKI_TYPE proposal (name + fields + sub_sections).
        Returns a dict like {"name": "...", "fields": [...], "sub_sections": [...]}
        and records it to schema_update_suggestions.json with approved:"pending".
        """
        prompt = f"""
You propose a new wiki type for documents that don't fit existing types.

Return ONLY JSON with keys: name, fields (list), sub_sections (list).

Document (truncated):
\"\"\"{doc_text[:4000]}\"\"\"
"""
        raw = self.llm.invoke(prompt)
        try:
            proposal = safe_json_loads(raw, {
                "wiki_type": "Research Study",
                "subtype": "Auto-generated",
                "description": f"Auto-generated from {file_name}",
                "fields": ["title", "description", "methodology", "findings"],
                "sub_sections": ["Key Insights", "Data Points", "Recommendations"]
            })
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                # Try to extract just the JSON part before the extra data
                try:
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(raw):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end > 0:
                        json_part = raw[:json_end]
                        proposal = safe_json_loads(json_part, {
                            "wiki_type": "Research Study",
                            "subtype": "Auto-generated",
                            "description": f"Auto-generated from {file_name}",
                            "fields": ["title", "description", "methodology", "findings"],
                            "sub_sections": ["Key Insights", "Data Points", "Recommendations"]
                        })
                    else:
                        # very defensive fallback
                        from pathlib import Path
                        guessed = Path(file_name).stem.replace("_"," ").replace("-"," ").strip()
                        proposal = {
                            "wiki_type": "Research Study",
                            "subtype": guessed,
                            "description": f"Auto-generated from {file_name}",
                            "fields": ["title", "description", "methodology", "findings"],
                            "sub_sections": ["Key Insights", "Data Points", "Recommendations"]
                        }
                except Exception:
                    # very defensive fallback
                    from pathlib import Path
                    guessed = Path(file_name).stem.replace("_"," ").replace("-"," ").strip()
                    proposal = {
                        "wiki_type": "Research Study",
                        "subtype": guessed,
                        "description": f"Auto-generated from {file_name}",
                        "fields": ["title", "description", "methodology", "findings"],
                        "sub_sections": ["Key Insights", "Data Points", "Recommendations"]
                    }
            else:
                # very defensive fallback
                from pathlib import Path
                guessed = Path(file_name).stem.replace("_"," ").replace("-"," ").strip()
                proposal = {
                    "wiki_type": "Research Study",
                    "subtype": guessed,
                    "description": f"Auto-generated from {file_name}",
                    "fields": ["title", "description", "methodology", "findings"],
                    "sub_sections": ["Key Insights", "Data Points", "Recommendations"]
                }
        except Exception:
            # very defensive fallback
            from pathlib import Path
            guessed = Path(file_name).stem.replace("_"," ").replace("-"," ").strip()
            proposal = {
                "name": f"Custom Study: {guessed}",
                "fields": ["title","date","methodology","coverage"],
                "sub_sections": ["Key Findings","Insights","Recommendations"]
            }

        # record suggestion with approved:"pending"
        self.record_schema_suggestion(proposal, file_name)
        print(f"💡 Proposed NEW wiki type: {proposal.get('name')} (approved: pending)")
        print(f"   ↳ recorded in schema_update_suggestions.json for file '{file_name}'")
        return proposal



    def classify_document(self, doc_text: str, top_k: int = DEFAULT_CLASSIFY_TOPK) -> Dict:
        """
        Returns:
        {
            "matches": [{"wiki_type": "...", "subtype": "..."|null, "confidence": 0.83}, ...],
            "new_type": null
        }
        """

        # Give model the full schema detail, not just type names
        types = list(self.schema.keys())
        schema_detail = {t: self.schema[t] for t in types}

        prompt = f"""
    You are a document classifier for a knowledge wiki.

    ## Existing wiki types (and structures):
    {json.dumps(schema_detail, indent=2)}

    ## Rules
    - You MUST assign the document to one of the existing wiki types above. 
    - Never invent or propose new types like "Custom Study".
    - If the document is some kind of research/analysis/diagnosis/report/etc., map it to "Research Study".
    - In that case, set "subtype" to a short descriptive string, e.g. "Brand Awareness", "Diagnosis", "Lore".
    - For all other docs, map to the closest generic type: Brand, Installment, Company, Studio, Person, Platform, Market.
    - Always return up to {top_k} best matches, with a confidence 0.0 - 1.0.
    - Do NOT include "new_type" unless no schema types exist (which is never the case here).
    - Answer ONLY JSON with keys: matches (list), new_type (must be null).

    ## Document (truncated):
    \"\"\"{doc_text[:4000]}\"\"\"
    """

        raw = self.llm.invoke(prompt)
        try:
            # Clean up the response - remove markdown code blocks if present
            cleaned = raw.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = safe_json_loads(cleaned, {"matches": [], "new_type": None})
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                # Try to extract just the JSON part before the extra data
                try:
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(cleaned):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end > 0:
                        json_part = cleaned[:json_end]
                        data = safe_json_loads(json_part, {"matches": [], "new_type": None})
                    else:
                        data = {"matches": [], "new_type": None}
                except Exception:
                    data = {"matches": [], "new_type": None}
            else:
                print(f"    ⚠️ JSON parsing failed for classify_document: {e}")
                print(f"    Raw response: {raw[:200]}...")
                data = {"matches": [], "new_type": None}
        except Exception as e:
            print(f"    ⚠️ JSON parsing failed for classify_document: {e}")
            print(f"    Raw response: {raw[:200]}...")
            data = {"matches": [], "new_type": None}

        matches = data.get("matches", [])
        for m in matches:
            m.setdefault("subtype", None)
            # Fix: LLM sometimes returns "type" instead of "wiki_type"
            if "type" in m and "wiki_type" not in m:
                m["wiki_type"] = m.pop("type")
            try:
                m["confidence"] = float(m.get("confidence", 0))
            except Exception:
                m["confidence"] = 0.0
        return {"matches": matches, "new_type": None}

    def record_schema_suggestion(self, new_type: Dict, file_name: str):
        """
        Append a pending suggestion if not already present.
        new_type = {"name": "...", "fields": [...], "sub_sections": [...]}
        """
        if not new_type or not new_type.get("name"):
            return
        # dedupe by name
        existing = {s["name"] for s in self.suggestions["suggestions"]}
        entry = {
            "name": new_type["name"],
            "fields": new_type.get("fields", []),
            "sub_sections": new_type.get("sub_sections", []),
            "approved": "pending",
            "candidate_files": [file_name]
        }
        if new_type["name"] not in existing:
            self.suggestions["suggestions"].append(entry)
        else:
            # attach file if new
            for s in self.suggestions["suggestions"]:
                if s["name"] == new_type["name"]:
                    if file_name not in s.get("candidate_files", []):
                        s.setdefault("candidate_files", []).append(file_name)
        self._save_suggestions()

    def apply_approved_suggestions(self) -> List[str]:
        """
        Move all suggestions with approved:"ok" into wiki_schema.json.
        Returns list of newly added type names.
        """
        added = []
        pending = []
        for s in self.suggestions["suggestions"]:
            if s.get("approved") == "ok":
                tname = s["name"]
                if tname not in self.schema:
                    self.schema[tname] = {
                        "fields": s.get("fields", []),
                        "sub_sections": s.get("sub_sections", []),
                        "links": []
                    }
                    added.append(tname)
            else:
                pending.append(s)
        if added:
            self._save_schema()
        self.suggestions["suggestions"] = pending
        self._save_suggestions()
        return added

    # ---------- wiki generation ----------
    def _yaml_front_matter(self, meta: Dict) -> str:
        """Generate clean YAML frontmatter compatible with Quartz."""
        lines = ["---"]
        
        # Standard Quartz fields with proper formatting
        if "title" in meta:
            lines.append(f'title: "{meta["title"]}"')
        
        if "subtitle" in meta and meta["subtitle"]:
            lines.append(f'subtitle: "{meta["subtitle"]}"')
        
        # Convert wiki_type to category
        if "wiki_type" in meta:
            lines.append(f'categories: ["{meta["wiki_type"]}"]')
        
        # Convert subtype to tags if it exists
        tags = []
        if "subtype" in meta and meta["subtype"]:
            tags.append(meta["subtype"])
        if "approved" in meta:
            tags.append(f"status:{meta['approved']}")
        
        if tags:
            lines.append(f'tags: {json.dumps(tags, ensure_ascii=False)}')
        
        # Add description if available
        if "description" in meta and meta["description"]:
            lines.append(f'description: "{meta["description"]}"')
        
        # Add source files as a note
        if "source_files" in meta and meta["source_files"]:
            lines.append(f'sources: {json.dumps(meta["source_files"], ensure_ascii=False)}')
        
        # Add schema version as internal metadata
        if "schema_version" in meta:
            lines.append(f'schema_version: {meta["schema_version"]}')
        
        lines.append("---\n")
        return "\n".join(lines)



    def _build_json_plan_prompt(
        self, wiki_type: str, subtype: Optional[str], schema: Dict,
        entity_name: str, source_file: str, doc_text: str
    ) -> str:
        fields = schema.get("fields", [])
        subs   = schema.get("sub_sections", [])
        return f"""
Emit ONLY JSON (no prose). Create a minimal content plan for a wiki page.

WIKI_TYPE: {wiki_type}
ENTITY: {entity_name}
ALLOWED_FIELDS: {json.dumps(fields, ensure_ascii=False)}
ALLOWED_SECTIONS: {json.dumps(subs, ensure_ascii=False)}

RULES:
- Use ONLY facts that appear in the SOURCE TEXT.
- Omit any field/section you cannot support. No placeholders (e.g., "unknown").
- Prefer 1–2 concise sentences per field or bullet points per section.
- Each item must cite the source as: "citations": [{{"doc":"{source_file}","chunk_id":"0"}}] (use dummy chunk '0' here).

OUTPUT SCHEMA:
{{
  "fields": {{ "<field>": {{"text":"…","citations":[{{"doc":"…","chunk_id":"…"}}]}} }},
  "sections": {{ "<section>": [{{"bullet":"…","citations":[{{"doc":"…","chunk_id":"…"}}]}}] }}
}}

SOURCE TEXT:
\"\"\"{doc_text[:6000]}\"\"\"
"""

    def _generate_markdown(self, wiki_type: str, subtype: Optional[str], entity_name: str, source_file: str, doc_text: str) -> str:
        # JSON-first → then render ourselves to ensure no empty placeholders
        print(f"🔍 _generate_markdown: Starting for {entity_name} ({wiki_type})")
        schema = self.schema.get(wiki_type, {"fields": [], "sub_sections": []})
        plan_prompt = self._build_json_plan_prompt(wiki_type, subtype, schema, entity_name, source_file, doc_text)
        print(f"🔍 _generate_markdown: Calling LLM for {entity_name}")
        raw = self.llm.invoke(plan_prompt)
        print(f"🔍 _generate_markdown: LLM response length: {len(raw)}")
        print(f"🔍 _generate_markdown: LLM response preview: {raw[:200]}...")
        try:
            from json_repair import repair_json
            print(f"🔍 _generate_markdown: Trying repair_json for {entity_name}")
            repaired = repair_json(raw)
            print(f"🔍 _generate_markdown: repair_json result length: {len(repaired)}")
            plan = safe_json_loads(repaired, {"fields": {}, "sections": {}})
        except Exception as e:
            print(f"🔍 _generate_markdown: repair_json failed for {entity_name}: {e}")
            try:
                print(f"🔍 _generate_markdown: Trying direct safe_json_loads for {entity_name}")
                plan = safe_json_loads(raw, {"fields": {}, "sections": {}})
            except json.JSONDecodeError as e:
                if "Extra data" in str(e):
                    # Try to extract just the JSON part before the extra data
                    try:
                        # Find the end of the JSON object
                        brace_count = 0
                        json_end = 0
                        for i, char in enumerate(raw):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        if json_end > 0:
                            json_part = raw[:json_end]
                            plan = safe_json_loads(json_part, {"fields": {}, "sections": {}})
                        else:
                            plan = {"fields": {}, "sections": {}}
                    except Exception:
                        plan = {"fields": {}, "sections": {}}
                else:
                    plan = {"fields": {}, "sections": {}}
            except Exception:
                plan = {"fields": {}, "sections": {}}

        def cite(cites):
            return " ".join([f"[^{source_file}:{c.get('chunk_id','0')}]" for c in (cites or [])])

        # Render minimal markdown
        md = [f"# {wiki_type}: {entity_name}\n"]
        for fld, obj in (plan.get("fields") or {}).items():
            if obj and obj.get("text"):
                md.append(f"**{fld}**  \n{obj['text']} {cite(obj.get('citations'))}\n")
        for sec, bullets in (plan.get("sections") or {}).items():
            if bullets:
                md.append(f"## {sec}")
                for b in bullets:
                    if b.get("bullet"):
                        md.append(f"- {b['bullet']} {cite(b.get('citations'))}")
        if (plan.get("fields") or {}) or (plan.get("sections") or {}):
            md.append("\n## References")
            md.append(f"[^{source_file}:0]: {source_file}")
        body_md = "\n".join(md)

        # Front matter
        front = self._yaml_front_matter({
            "wiki_type": wiki_type,
            "subtype": subtype or "",
            "title": entity_name,
            "approved": "pending" if wiki_type not in ("Brand", "Installment", "Research Study") else "ok",
            "source_files": [source_file],
            "schema_version": 1
        })
        return front + body_md.strip() + "\n"

    def _wiki_path(self, wiki_type: str, entity_name: str) -> str:
        slug = f"{_safe_slug(wiki_type)}_{_safe_slug(entity_name)}.md"
        return os.path.join(WIKI_FOLDER, slug)

    def create_or_update_wikis_from_matches(
        self, matches: List[Dict], entity_name: str, source_file: str, doc_text: str, file_hash: str
    ) -> List[str]:
        """
        For each matched wiki type above threshold, create/update the wiki.
        Also ensures (:WikiPage)-[:CITES]->(:Document).
        Returns list of written paths.
        """
        written = []
        for m in matches:
            if m.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                continue
            wiki_type = m["wiki_type"]
            subtype   = m.get("subtype")
            md = self._generate_markdown(wiki_type, subtype, entity_name, source_file, doc_text)
            path = self._wiki_path(wiki_type, entity_name)

            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            written.append(path)
            print(f"✅ Wiki written: {path}")

            # Link wiki -> Document
            driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
            )
            with driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {source_id:$source_id})
                    MERGE (w:WikiPage {path:$wiki_path})
                    MERGE (w)-[:CITES]->(d)
                    """,
                    source_id=file_hash,
                    wiki_path=path
                )
        return written

    def wiki_exists(self, wiki_type: str, entity_name: str) -> bool:
        path = self._wiki_path(wiki_type, entity_name)
        return os.path.exists(path)

    def ensure_entity_schema_basics(self):
        # Make sure common entity types exist (Brand, Company/Studio, Person, Platform, Market)
        changed = False
        defaults = {
            "Brand": {
                "fields": ["name","owner","first_release","key_values","tagline"],
                "sub_sections": ["Overview","Franchises/Installments","Brand Values","Audience","Notable Moments"],
                "links": []
            },
            "Company": {
                "fields": ["name","founded","hq","ceo","web","description"],
                "sub_sections": ["Studios","Games/Brands","Strategy","Financials","Leadership"],
                "links": [],
                "field_notes": {
                    "founded": "Usually TBD in research documents",
                    "hq": "Usually TBD in research documents", 
                    "ceo": "Usually TBD in research documents",
                    "web": "Usually TBD in research documents"
                }
            },
            "Studio": {
                "fields": ["name","parent_company","hq","notable_titles","leads"],
                "sub_sections": ["Teams","Projects","Culture","Hiring"],
                "links": []
            },
            "Person": {
                "fields": ["name","role","affiliation","bio"],
                "sub_sections": ["Career","Notable Work","Press Quotes"],
                "links": []
            },
            "Platform": {
                "fields": ["name","maker","generation","release_date"],
                "sub_sections": ["Hardware","Services","Installed Base"],
                "links": []
            },
            "Market": {
                "fields": ["name","region_code"],
                "sub_sections": ["Size","Trends","Competitors"],
                "links": []
            }
        }
        for k,v in defaults.items():
            if k not in self.schema:
                self.schema[k] = v
                changed = True
        if changed:
            self._save_schema()

    def _map_entity_to_wiki_type(self, etype: str) -> str:
        # normalize entity types to wiki types
        return {
            "brand": "Brand",
            "installment": "Installment",  # add this type to schema if you want its page
            "studio": "Studio",
            "company": "Company",
            "publisher": "Company",
            "person": "Person",
            "platform": "Platform",
            "market": "Market"
        }.get(etype, "Brand")


    def upsert_entity_wiki(self, entity: dict, doc_text: str, source_file: str, file_hash: str) -> str:
        """
        entity = {"name": "...", "type": "brand|company|studio|..."}
        - If wiki exists -> merge new facts (LLM guided).
        - Else -> create stub via LLM from schema.
        - Always link wiki -> Document in Neo4j using CITES edge.
        Returns path to the wiki.
        """
        print(f"🔍 upsert_entity_wiki: Processing entity {entity}")
        self.ensure_entity_schema_basics()

        wiki_type = self._map_entity_to_wiki_type(entity["type"])
        name = entity["name"]
        print(f"🔍 upsert_entity_wiki: Mapped to wiki_type: {wiki_type}, name: {name}")

        # 🟢 Canonicalize the entity name using Neo4j
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        with driver.session() as session:
            canonical = canonical_name_from_graph(name, wiki_type, session)
            if canonical != name:
                print(f"🔗 Using canonical name from graph: {canonical} (was: {name})")
            name = canonical

        path = self._wiki_path(wiki_type, name)
        schema = self.schema.get(wiki_type, {"fields": [], "sub_sections": []})

        if os.path.exists(path):
            # MERGE: ask LLM to update fields/subsections from new doc while preserving existing content
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
            prompt = f"""
You will update an existing wiki page using a new source. Return UPDATED MARKDOWN ONLY.

TARGET: {name} ({wiki_type})
NEW SOURCE: {source_file}

EXISTING WIKI PAGE:
{existing}

RULES:
- Add only facts explicitly present in NEW_SOURCE_TEXT (no world knowledge).
- Do not remove existing content unless the new source contradicts it, in which case replace and keep the rest.
- Do not add placeholders for missing info.
- Any new or changed sentence must include a footnote [^{source_file}:0].
- Keep the current structure and YAML front matter; only modify content fields/sections that have new facts.

NEW_SOURCE_TEXT:
\"\"\"{doc_text[:6000]}\"\"\"
"""
            updated = self.llm.invoke(prompt).strip()
            # Prune any legacy placeholders before writing
            updated = self._prune_placeholders(updated)
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated + "\n")
            print(f"🧩 Updated entity wiki: {path}")
        else:
            # CREATE: generate from schema + doc
            md = self._generate_markdown(wiki_type, None, name, source_file, doc_text)
            front = self._yaml_front_matter({
                "wiki_type": wiki_type,
                "title": name,
                "approved": "ok",  # entity pages are allowed immediately
                "source_files": [source_file],
                "schema_version": 1
            })
            with open(path, "w", encoding="utf-8") as f:
                f.write(front + md.strip() + "\n")
            print(f"🌱 Created entity wiki: {path}")

        # Always link wiki -> Document
        with driver.session() as session:
            session.run(
                """
                MATCH (d:Document {source_id:$source_id})
                MERGE (w:WikiPage {path:$wiki_path})
                MERGE (w)-[:CITES]->(d)
                """,
                source_id=file_hash,
                wiki_path=path
            )
            
            # 🔗 Create Brand-Installment cross-references
            self._create_brand_installment_cross_references(session, wiki_type, name, path)

        return path
    
    def _create_brand_installment_cross_references(self, session, wiki_type: str, entity_name: str, wiki_path: str):
        """
        Create cross-references between Brand and Installment wikis.
        This ensures bidirectional linking for Quartz navigation.
        """
        if wiki_type == "Brand":
            # Find all Installments that belong to this Brand
            result = session.run("""
                MATCH (b:Brand {name: $brand_name})<-[:BELONGS_TO_BRAND]-(i:Installment)
                RETURN i.name AS installment_name
            """, brand_name=entity_name)
            
            installments = [record["installment_name"] for record in result]
            if installments:
                # Add Installments section to Brand wiki
                self._add_installments_to_brand_wiki(entity_name, installments)
        
        elif wiki_type == "Installment":
            # Find the Brand this Installment belongs to
            result = session.run("""
                MATCH (i:Installment {name: $installment_name})-[:BELONGS_TO_BRAND]->(b:Brand)
                RETURN b.name AS brand_name
            """, installment_name=entity_name)
            
            brand_record = result.single()
            if brand_record:
                brand_name = brand_record["brand_name"]
                # Add Brand reference to Installment wiki
                self._add_brand_to_installment_wiki(entity_name, brand_name)
    
    def _add_installments_to_brand_wiki(self, brand_name: str, installments: List[str]):
        """
        Add Installments section to Brand wiki with Quartz-compatible links.
        """
        brand_wiki_path = self._wiki_path("Brand", brand_name)
        if not os.path.exists(brand_wiki_path):
            return
        
        # Read existing content
        with open(brand_wiki_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if Installments section already exists
        if "## Installments" in content:
            # Update existing section
            installments_links = []
            for installment in installments:
                installment_slug = f"Installment_{_safe_slug(installment)}"
                installments_links.append(f"- [[{installment_slug}|{installment}]]")
            
            # Replace existing installments list
            import re
            pattern = r"(## Installments\n)(.*?)(\n##|\n$)"
            replacement = f"\\1{chr(10).join(installments_links)}\\3"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        else:
            # Add new Installments section before the last section
            installments_links = []
            for installment in installments:
                installment_slug = f"Installment_{_safe_slug(installment)}"
                installments_links.append(f"- [[{installment_slug}|{installment}]]")
            
            installments_section = f"\n## Installments\n{chr(10).join(installments_links)}\n"
            
            # Insert before the last section (usually References or similar)
            lines = content.split('\n')
            insert_index = len(lines) - 2  # Before the last line
            lines.insert(insert_index, installments_section)
            content = '\n'.join(lines)
        
        # Write updated content
        with open(brand_wiki_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"🔗 Added {len(installments)} installments to {brand_name} wiki")
    
    def _add_brand_to_installment_wiki(self, installment_name: str, brand_name: str):
        """
        Add Brand reference to Installment wiki with Quartz-compatible link.
        """
        installment_wiki_path = self._wiki_path("Installment", installment_name)
        if not os.path.exists(installment_wiki_path):
            return
        
        # Read existing content
        with open(installment_wiki_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Add brand field if not present
        if "**brand:**" not in content.lower():
            # Find the fields section and add brand
            brand_slug = f"Brand_{_safe_slug(brand_name)}"
            brand_link = f"**brand:** [[{brand_slug}|{brand_name}]]"
            
            # Insert after the first field (usually name)
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("**name:**"):
                    lines.insert(i + 1, brand_link)
                    break
            
            content = '\n'.join(lines)
        
        # Write updated content
        with open(installment_wiki_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"🔗 Added brand reference to {installment_name} wiki")

    def _prune_placeholders(self, md: str) -> str:
        """
        Remove legacy placeholder lines from older pages.
        """
        lines = []
        for ln in md.splitlines():
            if re.search(r'\[No new information provided', ln, re.I):
                continue
            lines.append(ln)
        return re.sub(r'\n{3,}', '\n\n', "\n".join(lines)).strip() + "\n"

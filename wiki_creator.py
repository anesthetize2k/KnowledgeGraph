# wiki_creator.py
import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from litellm_wrapper import LiteLLMChat

load_dotenv()
from neo4j import GraphDatabase

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

    # ---------- schema & suggestions ----------
    def _load_schema(self) -> Dict:
        if not os.path.exists(WIKI_SCHEMA_PATH):
            raise FileNotFoundError(f"{WIKI_SCHEMA_PATH} not found. Please add it.")
        with open(WIKI_SCHEMA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("wiki_types", {})

    def _save_schema(self):
        with open(WIKI_SCHEMA_PATH, "w", encoding="utf-8") as f:
            json.dump({"wiki_types": self.schema}, f, indent=2, ensure_ascii=False)

    def _load_suggestions(self) -> Dict:
        if os.path.exists(SCHEMA_SUGGESTIONS_PATH):
            with open(SCHEMA_SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
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
            proposal = json.loads(raw)
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


    # ---------- classification ----------
    def classify_document(self, doc_text: str, top_k: int = DEFAULT_CLASSIFY_TOPK) -> Dict:
        """
        Returns:
          {
            "matches": [{"wiki_type": "...", "subtype": "..."|null, "confidence": 0.83}, ...],
            "new_type": {"name": "New Type", "fields": [...], "sub_sections": [...] } | null
          }
        """
        types = list(self.schema.keys())

        # If the schema includes Study Subtypes in a nested dict, surface them for guidance:
        subtypes = []
        if "Study Subtypes" in self.schema and isinstance(self.schema["Study Subtypes"], dict):
            subtypes = list(self.schema["Study Subtypes"].keys())

        prompt = f"""
You classify documents into wiki types and optionally subtypes.

Existing wiki types:
{types}

Existing study subtypes (if applicable):
{subtypes}

Rules:
- Return up to {top_k} best matches with confidence 0.0 - 1.0.
- If none fit well, propose NEW_WIKI_TYPE with a minimal structure (fields, sub_sections).
- If a match is a Study subtype, set wiki_type="Research Study" and include "subtype".
- Answer ONLY JSON with keys: matches (list), new_type (object or null).

Document (truncated):
\"\"\"{doc_text[:4000]}\"\"\"
"""
        raw = self.llm.invoke(prompt)
        try:
            data = json.loads(raw)
        except Exception:
            # very defensive fallback
            data = {"matches": [], "new_type": None}
        # sanitize
        matches = data.get("matches", [])
        for m in matches:
            m.setdefault("subtype", None)
            try:
                m["confidence"] = float(m.get("confidence", 0))
            except Exception:
                m["confidence"] = 0.0
        new_type = data.get("new_type", None)
        return {"matches": matches, "new_type": new_type}

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
        lines = ["---"]
        for k, v in meta.items():
            if isinstance(v, list):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            else:
                vv = str(v).replace("\n", " ").strip()
                lines.append(f"{k}: {vv}")
        lines.append("---\n")
        return "\n".join(lines)

    def _build_generation_prompt(self, wiki_type: str, subtype: Optional[str], schema: Dict, entity_name: str, source_file: str, doc_text: str) -> str:
        fields = schema.get("fields", [])
        subs   = schema.get("sub_sections", [])
        return f"""
You create Markdown wiki pages using the provided structure.

Wiki type: {wiki_type}
Subtype: {subtype or "None"}
Entity name (title): {entity_name}

Fields (fill concisely, make "TBD" or "Not specified in source document" if the document lacks data):
{fields}

IMPORTANT: For Company entities, these fields are commonly empty and should be "TBD":
- founded: Usually not mentioned in research documents
- hq: Usually not mentioned in research documents  
- ceo: Usually not mentioned in research documents
- web: Usually not mentioned in research documents

CRITICAL COMPANY FIELD RULES:
- **founded**: MUST be "TBD" unless the document explicitly states a founding date
- **hq**: MUST be "TBD" unless the document explicitly states a headquarters location  
- **ceo**: MUST be "TBD" unless the document explicitly states a CEO name
- **web**: MUST be "TBD" unless the document explicitly states a website URL
- **Studios section**: MUST be "Not specified in source document" unless specific studios are mentioned

REMEMBER: These fields are commonly empty in research documents and should default to "TBD" or "Not specified in source document" unless explicitly found in the text above.

Sub-sections (write clear bullet points when possible):
{subs}

Source file: {source_file}

Document (truncated):
\"\"\"{doc_text[:6000]}\"\"\"

CRITICAL RULES - READ CAREFULLY:
- Output valid Markdown.
- Top H1 must be "# {wiki_type}: {entity_name}".
- For fields, use bold labels like **Field_Name**: value
- For sub-sections, use "## {{Subsection}}" and content below.

STRICT FACTUAL REQUIREMENTS - ZERO TOLERANCE FOR HALLUCINATION:
- ONLY use information that is EXPLICITLY stated in the provided document text above.
- DO NOT add any information from your general knowledge, even if you think it's common knowledge.
- DO NOT infer or assume facts not directly stated in the document.
- If a field cannot be filled with information from the document, use "TBD" or "Not specified in source document".
- For company information (founded date, HQ, CEO, website, studio locations), ONLY include what is explicitly mentioned in the document.
- If you're unsure whether information comes from the document, DO NOT include it.

CRITICAL: The following fields MUST be "TBD" or "Not specified in source document" unless explicitly found in the document:
- founded date
- headquarters (hq)
- CEO name
- website URL
- studio locations
- any financial information
- any leadership details beyond what's explicitly stated

EXAMPLES OF WHAT NOT TO DO:
- DO NOT add founding dates, CEO names, or headquarters unless explicitly stated
- DO NOT add studio locations unless specifically mentioned
- DO NOT add generic company descriptions not found in the document
- DO NOT add industry rankings or leadership claims not stated

EXAMPLES OF WHAT TO DO:
- Use exact statistics, percentages, and numbers from the document
- Quote specific phrases and terminology from the document
- Reference specific studies, methodologies, and findings mentioned
- Include only the brands, franchises, and details explicitly listed
"""

    def _generate_markdown(self, wiki_type: str, subtype: Optional[str], entity_name: str, source_file: str, doc_text: str) -> str:
        schema = self.schema.get(wiki_type, {"fields": [], "sub_sections": []})
        prompt = self._build_generation_prompt(wiki_type, subtype, schema, entity_name, source_file, doc_text)
        body_md = self.llm.invoke(prompt)

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

    def create_or_update_wikis_from_matches(self, matches: List[Dict], entity_name: str, source_file: str, doc_text: str) -> List[str]:
        """
        For each matched wiki type above threshold, create/update the wiki.
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

            if os.path.exists(path):
                # merge source_files if needed; otherwise overwrite content to keep deterministic structure
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        existing = f.read()
                    # naive merge of source_files in front matter (best effort)
                    if "source_files:" in existing:
                        if source_file not in existing:
                            existing = existing.replace("source_files:", f"source_files: [{json.dumps(source_file)}]")
                except Exception:
                    pass

            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            written.append(path)
            print(f"✅ Wiki written: {path}")
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

    def upsert_entity_wiki(self, entity: dict, doc_text: str, source_file: str) -> str:
        """
        entity = {"name": "...", "type": "brand|company|studio|..."}
        - If wiki exists -> merge new facts (LLM guided).
        - Else -> create stub via LLM from schema.
        Returns path to the wiki.
        """
        self.ensure_entity_schema_basics()

        wiki_type = self._map_entity_to_wiki_type(entity["type"])
        name = entity["name"]

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
You are updating an existing wiki page.

Current page (Markdown with YAML front matter):
---
(omitted on purpose)
---
{existing}

New source file: {source_file}

Document excerpt:
\"\"\"{doc_text[:6000]}\"\"\"

CRITICAL TASK REQUIREMENTS:
- Keep existing structure and content.
- ONLY add/modify where the new document provides **new factual info** that is EXPLICITLY stated.
- Fill missing fields ONLY with information explicitly found in the new document.
- Append brief new bullets to matching sub-sections; don't duplicate.
- Add a **References** section at the end if not present, with a bullet for the new source file.

STRICT FACTUAL REQUIREMENTS:
- DO NOT add any information from your general knowledge, even if it seems relevant.
- DO NOT infer or assume facts not directly stated in the new document.
- ONLY use information that is EXPLICITLY mentioned in the provided document excerpt above.
- If you're unsure whether information comes from the document, DO NOT include it.

- Output the full updated Markdown.
"""
            updated = self.llm.invoke(prompt).strip()
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated + "\n")
            print(f"🧩 Updated entity wiki: {path}")
            return path

        # CREATE: generate from schema + doc
        gen_prompt = self._build_generation_prompt(
            wiki_type=wiki_type, subtype=None, schema=schema,
            entity_name=name, source_file=source_file, doc_text=doc_text
        )
        md = self.llm.invoke(gen_prompt)
        front = self._yaml_front_matter({
            "wiki_type": wiki_type, "title": name, "approved": "ok",
            "source_files": [source_file], "schema_version": 1
        })
        with open(path, "w", encoding="utf-8") as f:
            f.write(front + md.strip() + "\n")
        print(f"🌱 Created entity wiki: {path}")
        return path

# triplet_extractor.py
import json, re, os
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from litellm_wrapper import LiteLLMChat
from ontology_expander import OntologyExpander

load_dotenv()

# Allowed by your ontology
ALLOWED_TYPES = {
    "brand","installment","studio","publisher","platform","market",
    "player_segment","brand_metric","research_study","time_period",
    "insight","competitor","metric_value","event","concept","feature",
    "brand_characteristic","person","company","expansion","content",
    "gameplay_feature","metric","region","audience_type","policy",
    "ministry","date"
}

ALLOWED_RELS = {
    "belongs_to_brand","developed_by","published_by","released_on",
    "measured_by","measured_for","collected_in","covers_period",
    "targets_segment","analyzes","compares_with","has_insight","documented_in",
    "is_type_of","is_synonym_of","is_feature_of","headed_by","announced",
    "drives","appeals_to","includes","performs_on","targets","has_minister",
    "start_date"
}

# Load glossary for canonical name resolution
def load_glossary():
    """Load glossary with synonyms for canonical name resolution."""
    try:
        with open("glossary.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load glossary.json: {e}")
        return {"synonyms": {}}

GLOSSARY = load_glossary()

# Canonical synonyms / normalizer
ALIASES = {}
for canonical_name, synonyms in GLOSSARY.get("synonyms", {}).items():
    for synonym in synonyms:
        ALIASES[synonym.lower()] = canonical_name

GENERIC_BAD = {
    "these elements","brand","element","elements","topic","section",
    "this","that","it","intro","summary","n/a","none",""
}

PROMPT = """You are an information extraction model.

Extract FACTUAL 5-tuples from the passage as:
[head, head_type, relation, tail, tail_type]

Rules:
- head_type/tail_type MUST be one of:
  brand, installment, studio, publisher, platform, market, player_segment,
  brand_metric, research_study, time_period, insight, competitor, metric_value,
  event, concept, feature, brand_characteristic, person, company, expansion,
  content, gameplay_feature, metric, region, audience_type, policy, ministry, date
- relation MUST be one of:
  belongs_to_brand, developed_by, published_by, released_on,
  measured_by, measured_for, collected_in, covers_period,
  targets_segment, analyzes, compares_with, has_insight, documented_in,
  is_type_of, is_synonym_of, is_feature_of, headed_by, announced,
  drives, appeals_to, includes, performs_on, targets, has_minister, start_date
- Use canonical names (e.g., "Assassin's Creed", "PlayStation 5").
- Ignore vague fragments like "these elements", "brand", "topic".
- Dates go into time_period (e.g., "Q1 2024", "June 2025").
- Numbers/percentages go into metric_value (with unit if present).
- Output ONLY a JSON list of 5-tuples. No prose.

Examples:
Input: "The Ministry of Finance, headed by Nirmala Sitharaman, announced the Fiscal Responsibility Act on 1 Feb 2024."
Output:
[
  ["Ministry of Finance","ministry","has_minister","Nirmala Sitharaman","person"],
  ["Ministry of Finance","ministry","announced","Fiscal Responsibility Act","policy"],
  ["Fiscal Responsibility Act","policy","start_date","1 Feb 2024","date"]
]

Input:
"Assassin's Creed Shadows is developed by Ubisoft Quebec and released on PS5 and PC in 2025."
Output:
[
  ["Assassin's Creed Shadows","installment","belongs_to_brand","Assassin's Creed","brand"],
  ["Assassin's Creed Shadows","installment","developed_by","Ubisoft Quebec","studio"],
  ["Assassin's Creed Shadows","installment","released_on","PlayStation 5","platform"],
  ["Assassin's Creed Shadows","installment","released_on","PC","platform"],
  ["Assassin's Creed Shadows","installment","covers_period","2025","time_period"]
]

Input:
"The document mentions AC (short for Assassin's Creed) and Ubi (Ubisoft's nickname)."
Output:
[
  ["AC","brand","is_synonym_of","Assassin's Creed","brand"],
  ["Ubi","company","is_synonym_of","Ubisoft","company"]
]

Passage:
\"\"\"{text}\"\"\""""

def _canon(s: str) -> str:
    if not s: return ""
    t = re.sub(r"\s+", " ", s.strip().strip("-–—·•:")).strip()
    low = t.lower()
    if low in ALIASES:
        return ALIASES[low]
    # strip ™ ® quotes
    t = t.replace("™","").replace("®","").strip("'\"\"")
    return t

def _is_bad(s: str) -> bool:
    return s.lower() in GENERIC_BAD or len(s.strip()) < 2

def _valid_type(t: str) -> bool:
    return t in ALLOWED_TYPES

def _valid_rel(r: str) -> bool:
    return r in ALLOWED_RELS

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

class TripletExtractor:
    def __init__(self):
        self.llm = LiteLLMChat(model="gpt-4o-mini")  # Explicitly use mini for triplet extraction
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def extract(self, text: str) -> List[Tuple[str,str,str,str,str]]:
        out = self.llm.invoke(PROMPT.format(text=text[:3500]))
        try:
            data = json.loads(out)
        except Exception:
            return []

        cleaned = []
        for tup in data:
            if not (isinstance(tup, list) and len(tup) == 5):
                continue
            h, ht, r, t, tt = tup
            h, t = _canon(str(h)), _canon(str(t))
            ht, tt, r = str(ht), str(tt), str(r)

            if _is_bad(h) or _is_bad(t):            # drop generic junk
                continue
            if not _valid_type(ht) or not _valid_type(tt):
                continue
            if not _valid_rel(r):
                continue

            cleaned.append((h, ht, r, t, tt))
        return cleaned

    def upsert(self, triples: List[Tuple[str,str,str,str,str]], source_document_id: str = None, chunk_id: str = None):
        if not triples: return
        with self.driver.session() as sess:
            for h, ht, r, t, tt in triples:
                # Use canonical name resolution for both head and tail
                canonical_h = canonical_name_from_graph(h, ht.capitalize(), sess)
                canonical_t = canonical_name_from_graph(t, tt.capitalize(), sess)
                
                # Write with label guard + MERGE by (label, name)
                cypher = f"""
                MERGE (h:{ht.capitalize()} {{name: $h}})
                MERGE (t:{tt.capitalize()} {{name: $t}})
                MERGE (h)-[rel:{r}]->(t)
                """
                
                # Link entities to source document if provided
                if source_document_id:
                    cypher += """
                    WITH h, t, rel
                    MATCH (d:Document {source_id: $source_id})
                    MERGE (h)-[:MENTIONED_IN]->(d)
                    MERGE (t)-[:MENTIONED_IN]->(d)
                    """
                
                # Link entities to specific chunk if provided
                if chunk_id:
                    cypher += """
                    WITH h, t, rel
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (h)-[:MENTIONED_IN_CHUNK]->(c)
                    MERGE (t)-[:MENTIONED_IN_CHUNK]->(c)
                    """
                
                cypher += "RETURN id(rel) as id"
                
                try:
                    params = {"h": canonical_h, "t": canonical_t}
                    if source_document_id:
                        params["source_id"] = source_document_id
                    if chunk_id:
                        params["chunk_id"] = chunk_id
                    sess.run(cypher, **params)
                except Exception as e:
                    print(f"⚠️ Neo4j write failed for {canonical_h}-{r}->{canonical_t}: {e}")
            
            # 🔗 Ensure Brand-Installment connections
            self._ensure_brand_installment_connections(sess)
    
    def _ensure_brand_installment_connections(self, session):
        """
        Automatically create Brand-Installment connections based on naming patterns.
        This ensures that every Installment is connected to its Brand.
        """
        # Find all Installment nodes that don't have a BELONGS_TO_BRAND relationship
        result = session.run("""
            MATCH (i:Installment)
            WHERE NOT EXISTS((i)-[:BELONGS_TO_BRAND]->())
            RETURN i.name AS installment_name
        """)
        
        for record in result:
            installment_name = record["installment_name"]
            
            # Determine the brand based on the installment name
            brand_name = self._extract_brand_from_installment(installment_name)
            
            if brand_name:
                # Create the Brand node and connection
                session.run("""
                    MERGE (b:Brand {name: $brand_name})
                    MERGE (i:Installment {name: $installment_name})
                    MERGE (i)-[:BELONGS_TO_BRAND]->(b)
                """, brand_name=brand_name, installment_name=installment_name)
                print(f"🔗 Connected {installment_name} to {brand_name}")
    
    def _extract_brand_from_installment(self, installment_name: str) -> str:
        """
        Extract brand name from installment name using common patterns.
        """
        name_lower = installment_name.lower()
        
        # Assassin's Creed patterns
        if "assassin" in name_lower or "creed" in name_lower:
            return "Assassin's Creed"
        
        # Add more brand patterns here as needed
        # For example:
        # if "fifa" in name_lower:
        #     return "FIFA"
        # if "far cry" in name_lower:
        #     return "Far Cry"
        
        return None
    


    def process_chunks(self, chunks: List, source_document_id: str = None):
        print(f"    📊 Processing {len(chunks)} chunks for knowledge extraction...")
        total_triples = 0
        all_triples = []  # Collect all triples for ontology analysis
        
        for i, ch in enumerate(chunks):
            txt = ch.page_content
            triples = self.extract(txt)
            if triples:
                # Get chunk_id from chunk metadata
                chunk_id = ch.metadata.get("chunk_id") if hasattr(ch, 'metadata') else None
                print(f"      • chunk {i+1}/{len(chunks)}: {len(triples)} triples extracted")
                self.upsert(triples, source_document_id, chunk_id)
                total_triples += len(triples)
                
                # Collect triples for ontology analysis
                for triple in triples:
                    all_triples.append({
                        "head": triple[0],
                        "head_type": triple[1],
                        "relation": triple[2],
                        "tail": triple[3],
                        "tail_type": triple[4]
                    })
            else:
                print(f"      • chunk {i+1}/{len(chunks)}: no triples found")
        
        print(f"    ✅ Total triples extracted: {total_triples}")
        
        # Analyze ontology gaps if we have triples
        if all_triples:
            self._analyze_ontology_gaps(all_triples)
    
    def _analyze_ontology_gaps(self, triples: List[Dict]):
        """Analyze triples for ontology gaps and suggest new types."""
        try:
            expander = OntologyExpander()
            suggestions = expander.analyze_triple_gaps(triples)
            
            if suggestions:
                print(f"    🔍 Found {len(suggestions)} potential ontology gaps:")
                for suggestion in suggestions:
                    print(f"      • {suggestion['type']}: {suggestion['original_type']} → {suggestion['canonical_name']} (confidence: {suggestion['confidence']:.2f})")
                    expander.add_suggestion(suggestion)
                
                print(f"    📝 Suggestions saved to ontology_suggestions.json")
                print(f"    💡 Review and approve suggestions using the ontology management interface")
            else:
                print(f"    ✅ No ontology gaps detected")
                
        except Exception as e:
            print(f"    ⚠️ Ontology gap analysis failed: {e}")

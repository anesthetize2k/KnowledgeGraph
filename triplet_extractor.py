# triplet_extractor.py
import json, re, os
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from litellm_wrapper import LiteLLMChat

load_dotenv()

# Allowed by your ontology
ALLOWED_TYPES = {
    "brand","installment","studio","publisher","platform","market",
    "player_segment","brand_metric","research_study","time_period",
    "insight","competitor","metric_value"
}

ALLOWED_RELS = {
    "belongs_to_brand","developed_by","published_by","released_on",
    "measured_by","measured_for","collected_in","covers_period",
    "targets_segment","analyzes","compares_with","has_insight","documented_in"
}

# Canonical synonyms / normalizer
ALIASES = {
    # common brand aliases
    "ac": "Assassin's Creed",
    "assassin’s creed": "Assassin's Creed",
    "assassins creed": "Assassin's Creed",
    "assassin's creed": "Assassin's Creed",

    # platforms
    "ps5": "PlayStation 5",
    "xbsx": "Xbox Series X|S",
}

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
  brand_metric, research_study, time_period, insight, competitor, metric_value
- relation MUST be one of:
  belongs_to_brand, developed_by, published_by, released_on,
  measured_by, measured_for, collected_in, covers_period,
  targets_segment, analyzes, compares_with, has_insight, documented_in
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
"Assassin’s Creed Shadows is developed by Ubisoft Quebec and released on PS5 and PC in 2025."
Output:
[
  ["Assassin's Creed Shadows","installment","belongs_to_brand","Assassin's Creed","brand"],
  ["Assassin's Creed Shadows","installment","developed_by","Ubisoft Quebec","studio"],
  ["Assassin's Creed Shadows","installment","released_on","PlayStation 5","platform"],
  ["Assassin's Creed Shadows","installment","released_on","PC","platform"],
  ["Assassin's Creed Shadows","installment","covers_period","2025","time_period"]
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
    t = t.replace("™","").replace("®","").strip("'\"“”")
    return t

def _is_bad(s: str) -> bool:
    return s.lower() in GENERIC_BAD or len(s.strip()) < 2

def _valid_type(t: str) -> bool:
    return t in ALLOWED_TYPES

def _valid_rel(r: str) -> bool:
    return r in ALLOWED_RELS

class TripletExtractor:
    def __init__(self):
        self.llm = LiteLLMChat()
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

    def upsert(self, triples: List[Tuple[str,str,str,str,str]]):
        if not triples: return
        with self.driver.session() as sess:
            for h, ht, r, t, tt in triples:
                # Write with label guard + MERGE by (label, name)
                cypher = f"""
                MERGE (h:{ht.capitalize()} {{name: $h}})
                MERGE (t:{tt.capitalize()} {{name: $t}})
                MERGE (h)-[rel:{r}]->(t)
                RETURN id(rel) as id
                """
                try:
                    sess.run(cypher, h=h, t=t)
                except Exception as e:
                    print(f"⚠️ Neo4j write failed for {h}-{r}->{t}: {e}")

    def process_chunks(self, chunks: List):
        for i, ch in enumerate(chunks):
            txt = ch.page_content
            triples = self.extract(txt)
            if triples:
                print(f"   • chunk {i}: {len(triples)} triples")
                self.upsert(triples)

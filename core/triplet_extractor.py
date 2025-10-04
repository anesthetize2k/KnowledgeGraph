# triplet_extractor.py
import json, re, os, hashlib
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from .litellm_wrapper import LiteLLMChat
from .ontology_expander import OntologyExpander

load_dotenv()

# =========================
# 🎯 Tight Ontology (RAG-first)
# =========================
# Intentionally lean: only classes you’ll traverse or facet by.
ALLOWED_TYPES = {
    # Games & org
    "brand", "installment", "studio", "publisher", "company", "competitor",
    # Player & market
    "player_segment", "audience_type", "region", "market",
    # Tech/content/time
    "platform", "content", "gameplay_feature", "feature", "update", "event",
    "time_period", "date",
    # Research & insights (kept minimal)
    "research_report", "insight", "dataset",
    # People (for authors/PMs if needed)
    "person",
    # Metrics & observations
    "metric", "metric_value",
}

# Relations limited to those used in traversal & RAG enrichment
ALLOWED_RELS = {
    # Hierarchy / composition
    "belongs_to_brand", "part_of", "includes", "has_feature", "is_synonym_of",
    # Org & launch
    "developed_by", "published_by", "released_on", "available_on",
    # Scoping
    "targets_segment", "targets_audience", "for_region",
    # Research linkage & provenance
    "documents", "documented_in", "mentions", "derived_from",
    # Events / temporal
    "updated_by", "occurred_on", "during_period", "covers_period",
    # Metrics (ONLY these drive numeric enrichment)
    "has_metric", "has_metric_value", "of_metric", "measured_for",
    # Keep light causal terms (optional; use carefully)
    "impacts", "correlates_with",
    # Chunk provenance
    "mentioned_in_doc", "mentioned_in_chunk",
}

# =========================
# 🧠 Metric Canonicalization (tight + explicit)
# =========================
# Only metrics that materially enrich answers; everything else is dropped.
METRIC_WHITELIST = {
    # Engagement/retention
    "dau", "mau", "wau", "retention_d1", "retention_d7", "retention_d30",
    "churn_rate", "playtime", "avg_session_length", "sessions_per_user",
    # Monetization
    "arpdau", "arppu", "conversion_rate", "ltv", "payers_rate",
    # QoE
    "nps", "csat", "crash_rate", "bug_rate", "matchmaking_time", "queue_time",
    # Perf
    "sales", "revenue", "peak_ccu", "avg_ccu", "reviews_score",
    # Segmentation
    "segment_share",  # share of target segment within a population segment
}

# Common synonyms → canonical metric key (lower_snake_case)
METRIC_SYNONYMS = {
    "daily_active_users": "dau",
    "d.a.u.": "dau",
    "monthly_active_users": "mau",
    "day_1_retention": "retention_d1",
    "d1_retention": "retention_d1",
    "day_7_retention": "retention_d7",
    "d7_retention": "retention_d7",
    "day_30_retention": "retention_d30",
    "d30_retention": "retention_d30",
    "avg_session_duration": "avg_session_length",
    "session_duration": "avg_session_length",
    "average_session_length": "avg_session_length",
    "play_time": "playtime",
    "payer_rate": "payers_rate",
    "payers%": "payers_rate",
    "user_conversion": "conversion_rate",
    "mmr_time": "matchmaking_time",
    "queue_duration": "queue_time",
    "metacritic": "reviews_score",
    # shares
    "share": "segment_share",
    "proportion": "segment_share",
}

# =========================
# 🔤 Glossary (synonyms)
# =========================
def load_glossary():
    """Load glossary with synonyms for canonical name resolution (optional)."""
    try:
        with open("glossary.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load glossary.json: {e}")
        return {"synonyms": {}}

GLOSSARY = load_glossary()

ALIASES: Dict[str, str] = {}
for canonical_name, synonyms in GLOSSARY.get("synonyms", {}).items():
    for synonym in synonyms:
        ALIASES[synonym.lower()] = canonical_name

# =========================
# 🚯 Noise guards
# =========================
GENERIC_BAD = {
    "these elements","element","elements","topic","section","the study",
    "this","that","it","intro","summary","n/a","none","", "-", "—", "•"
}

# =========================
# 🧾 MUCH TIGHTER Prompt (Graph-RAG)
# =========================
PROMPT = """
You extract high-signal facts to build Ubisoft’s internal knowledge graph for games, player behavior, and performance.

OUTPUT STRICTLY AS JSON with keys: "entities" and "relations". No prose.

ENTITY SCHEMA
  {{ "name": str, "type": one_of[
      brand, installment, studio, publisher, company, competitor,
      player_segment, audience_type, region, market,
      platform, content, gameplay_feature, feature, update, event,
      time_period, date,
      research_report, insight, dataset,
      person,
      metric, metric_value
    ],
    "aliases": [str]? }}

RELATION SCHEMA
  {{
    "head": str, "head_type": <type>,
    "relation": one_of[
      belongs_to_brand, part_of, includes, has_feature, is_synonym_of,
      developed_by, published_by, released_on, available_on,
      targets_segment, targets_audience, for_region,
      documents, documented_in, mentions, derived_from,
      updated_by, occurred_on, during_period, covers_period,
      has_metric, has_metric_value, of_metric, measured_for,
      impacts, correlates_with,
      mentioned_in_doc, mentioned_in_chunk
    ],
    "tail": str, "tail_type": <type>,
    "qualifiers": {{
        "value": str|number?,     // numeric value ONLY if fully scoped (see rules)
        "unit": str?,             // %, hrs, mins, etc.
        "time_period": str?,      // Q1 2025, 2024-06, 2025
        "date": str?,             // 2024-11-28
        "segment": str?,          // player segment name (target)
        "population_segment": str?, // REQUIRED for segment_share (population)
        "region": str?, "platform": str?,
        "confidence": number?     // 0..1
    }}?
  }}

HARD RULES (DROP IF VIOLATED)
1) NO BARE NUMBERS: Never emit a numeric value without the metric name and scope. If you only see “11%”, do not produce a metric relation.
2) METRICS REQUIRE CONTEXT: Only emit metric relations if:
   - metric ∈ {{dau,mau,wau,retention_d1,retention_d7,retention_d30,churn_rate,playtime,avg_session_length,sessions_per_user,
               arpdau,arppu,conversion_rate,ltv,payers_rate,nps,csat,crash_rate,bug_rate,matchmaking_time,queue_time,
               sales,revenue,peak_ccu,avg_ccu,reviews_score,segment_share}}
   - AND at least one of: time_period/date/segment/region/platform is present.
3) SEGMENT SHARE SPECIALIZATION:
   - For metric=segment_share, qualifiers MUST include both:
     a) segment = <TARGET segment>, and
     b) population_segment = <POPULATION segment>.
   - Example: “11% of HD players are AC Casual players” ⇒ segment="AC Casual players", population_segment="HD players".
4) DATES/PERIODS ARE QUALIFIERS: Do not create standalone date entities unless they’re named periods (e.g., “Q1 2025”). Prefer qualifiers.
5) MENTIONS OVER ASSERTIONS: If context is insufficient to satisfy rules, output only entities and “mentions” relations from the chunk (no numbers).

EXTRACTION PRIORITY (KEEP ONLY HIGH-SIGNAL)
- Brand ↔ Installment topology; studio/publisher; release/availability platforms; updates/events tied to games/periods.
- Player segments, regions, and their links to games or metrics.
- Metrics WITH qualifiers (period/segment/region/platform). Drop generic/unscoped stats.

FORMAT
- Output ONLY: {{"entities":[...], "relations":[...]}}. If nothing valid, return {{"entities":[], "relations":[]}}.
- English only.

Passage:
\"\"\"{text}\"\"\"
""".strip()

# =========================
# 🔧 Helpers
# =========================
def _canon(s: Optional[str]) -> str:
    if not s:
        return ""
    t = re.sub(r"\s+", " ", s.strip().strip("-–—·•:")).strip()
    low = t.lower()
    if low in ALIASES:
        return ALIASES[low]
    t = t.replace("™", "").replace("®", "").strip("'\"\"")
    return t

def _is_bad(s: str) -> bool:
    return s.lower() in GENERIC_BAD or len(s.strip()) < 2

def _valid_type(t: str) -> bool:
    return t in ALLOWED_TYPES

def _valid_rel(r: str) -> bool:
    return r in ALLOWED_RELS

def _metric_canon(name: str) -> Optional[str]:
    low = name.strip().lower().replace("-", "_").replace(" ", "_")
    if low in METRIC_SYNONYMS:
        low = METRIC_SYNONYMS[low]
    return low if low in METRIC_WHITELIST else None

def _parse_numeric(value: Any) -> (Optional[float], Optional[str]):
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    s = str(value).strip()
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$", s)
    if m:
        return float(m.group(1)), "%"
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)\s*$", s, re.I)
    if m:
        return float(m.group(1)), "hrs"
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(m|min|mins|minute|minutes)\s*$", s, re.I)
    if m:
        return float(m.group(1)), "mins"
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1)), None
    return None, None

def _hash_observation(head: str, metric: str, val: Optional[float], unit: Optional[str],
                      period: Optional[str], segment: Optional[str], population_segment: Optional[str],
                      region: Optional[str], platform: Optional[str]) -> str:
    base = json.dumps({
        "head": head, "metric": metric, "val": val, "unit": unit,
        "period": period, "segment": segment, "population_segment": population_segment,
        "region": region, "platform": platform
    }, sort_keys=True)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]

def canonical_name_from_graph(name: str, label: str, session) -> str:
    rec = session.run(
        f"MATCH (n:{label}) WHERE toLower(n.name) = toLower($name) RETURN n.name AS name",
        name=name
    ).single()
    return rec["name"] if rec else name

def _label_for(t: str) -> str:
    return "".join(part.capitalize() for part in t.split("_"))

# =========================
# 🧩 Extraction + Cleaning
# =========================
class TripletExtractor:
    def __init__(self):
        self.llm = LiteLLMChat(model="gpt-4o-mini")
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()

    # ---------- LLM call ----------
    def _llm_extract(self, text: str) -> Dict[str, Any]:
        out = self.llm.invoke(PROMPT.format(text=text[:4000]))
        try:
            # Try to clean the response if it has extra data
            if "Extra data:" in str(out):
                # Extract JSON part before the error
                json_start = out.find('{')
                if json_start != -1:
                    out = out[json_start:]
                    # Find the end of the JSON object
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(out):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end > 0:
                        out = out[:json_end]
            
            parsed = json.loads(out)
            if isinstance(parsed, list):
                return {"entities": [], "relations": [
                    {"head": h, "head_type": ht, "relation": r, "tail": t, "tail_type": tt}
                    for (h, ht, r, t, tt) in parsed if isinstance(h, str)
                ]}
            if "entities" in parsed and "relations" in parsed:
                return parsed
        except Exception as e:
            print(f"⚠️ LLM parse failure: {e}")
            # Try multiple fallback strategies
            try:
                # Strategy 1: Try to extract JSON from the response
                if "{" in str(out) and "}" in str(out):
                    # Find the first complete JSON object
                    start = str(out).find('{')
                    if start != -1:
                        # Count braces to find the end
                        brace_count = 0
                        end = start
                        for i, char in enumerate(str(out)[start:], start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = i + 1
                                    break
                        if end > start:
                            json_str = str(out)[start:end]
                            parsed = json.loads(json_str)
                            if "entities" in parsed and "relations" in parsed:
                                return parsed
            except:
                pass
            
            try:
                # Strategy 2: Try to extract just entities and relations
                if "entities" in str(out) and "relations" in str(out):
                    # Find entities array
                    entities_start = str(out).find('"entities"')
                    if entities_start != -1:
                        # Find the end of entities array
                        entities_end = str(out).find(']', entities_start)
                        if entities_end != -1:
                            # Find relations array
                            relations_start = str(out).find('"relations"', entities_end)
                            if relations_start != -1:
                                relations_end = str(out).find(']', relations_start)
                                if relations_end != -1:
                                    # Construct minimal JSON
                                    json_str = '{"entities":' + str(out)[entities_start+11:entities_end+1] + ',"relations":' + str(out)[relations_start+12:relations_end+1] + '}'
                                    parsed = json.loads(json_str)
                                    if "entities" in parsed and "relations" in parsed:
                                        return parsed
            except:
                pass
                
        return {"entities": [], "relations": []}

    # ---------- Public APIs ----------
    def extract(self, text: str) -> List[Tuple[str, str, str, str, str]]:
        raw = self._llm_extract(text)
        entities = self._clean_entities(raw.get("entities", []))
        relations = self._clean_relations(raw.get("relations", []), entities_by_name={e["name"]: e for e in entities})
        return [(r["head"], r["head_type"], r["relation"], r["tail"], r["tail_type"]) for r in relations]

    def extract_structured(self, text: str) -> Dict[str, Any]:
        raw = self._llm_extract(text)
        entities = self._clean_entities(raw.get("entities", []))
        relations = self._clean_relations(raw.get("relations", []), entities_by_name={e["name"]: e for e in entities})
        return {"entities": entities, "relations": relations}

    # ---------- Cleaning ----------
    def _clean_entities(self, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned, seen = [], set()
        for e in ents:
            name = _canon(e.get("name", ""))
            t = e.get("type", "")
            if _is_bad(name) or not _valid_type(t):
                continue
            key = (name.lower(), t)
            if key in seen:
                continue
            seen.add(key)
            aliases = [a for a in (e.get("aliases") or []) if a and not _is_bad(a)]
            cleaned.append({"name": name, "type": t, "aliases": aliases})
        return cleaned

    def _relation_importance(self, r: Dict[str, Any]) -> float:
        rel = r.get("relation", "")
        ht, tt = r.get("head_type", ""), r.get("tail_type", "")
        q = r.get("qualifiers") or {}
        base = 0.0
        if rel in {"has_metric", "has_metric_value", "measured_for", "of_metric"}:
            base += 0.7
            if any(k in q for k in ("time_period", "date", "segment", "population_segment", "region", "platform")):
                base += 0.2
            if "value" in q:
                base += 0.1
        if rel in {"belongs_to_brand", "developed_by", "published_by", "released_on",
                   "available_on", "includes", "has_feature",
                   "targets_segment", "for_region", "during_period", "covers_period",
                   "updated_by", "occurred_on"}:
            base += 0.5
        if rel in {"documents", "documented_in", "mentions", "derived_from"}:
            base += 0.35
        if ht not in ALLOWED_TYPES or tt not in ALLOWED_TYPES:
            base -= 0.6
        return max(0.0, min(1.0, base))

    def _metrics_policy_ok(self, rel: str, ht: str, tt: str, q: Dict[str, Any], h: str, t: str) -> bool:
        """
        Enforce HARD RULES:
        - No bare numbers.
        - Metrics must be whitelisted + have context.
        - segment_share must have both segment & population_segment.
        """
        # If no metric relation, nothing to enforce here
        if rel not in {"has_metric", "has_metric_value", "measured_for", "of_metric"}:
            return True

        # Identify which side is the metric
        metric_name = None
        if tt == "metric":
            metric_name = t
        elif ht == "metric":
            metric_name = h

        metric_key = _metric_canon(metric_name) if metric_name else None
        if not metric_key:
            return False  # not a whitelisted metric

        # No bare numbers: if a value exists, require context
        has_value = "value" in q
        has_context = any(k in q for k in ("time_period", "date", "segment", "population_segment", "region", "platform"))
        if has_value and not has_context:
            return False

        # segment_share requires both segment (target) and population_segment
        if metric_key == "segment_share":
            if not (q.get("segment") and q.get("population_segment")):
                return False

        return True

    def _clean_relations(self, rels: List[Dict[str, Any]], entities_by_name: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = []
        for r in rels:
            h = _canon(str(r.get("head", "")))
            t = _canon(str(r.get("tail", "")))
            ht, tt = str(r.get("head_type", "")), str(r.get("tail_type", ""))
            rel = str(r.get("relation", ""))
            if _is_bad(h) or _is_bad(t): continue
            if not _valid_type(ht) or not _valid_type(tt): continue
            if not _valid_rel(rel): continue

            q = r.get("qualifiers") or {}

            # Canonicalize metrics (drop non-whitelisted)
            if tt == "metric":
                canon = _metric_canon(t)
                if not canon: continue
                t = canon
            if ht == "metric":
                canon = _metric_canon(h)
                if not canon: continue
                h = canon

            # Canonicalize metric name if present in qualifiers
            if isinstance(q.get("metric"), str):
                mcanon = _metric_canon(q["metric"])
                if not mcanon: continue
                q["metric"] = mcanon

            # Parse numeric (keep unit)
            if "value" in q:
                val, unit = _parse_numeric(q["value"])
                if val is not None:
                    q["value"] = val
                    if unit and not q.get("unit"):
                        q["unit"] = unit

            # Enforce hard metric policy
            if not self._metrics_policy_ok(rel, ht, tt, q, h, t):
                continue

            importance = self._relation_importance({"relation": rel, "head_type": ht, "tail_type": tt, "qualifiers": q})
            if importance < 0.5:
                continue

            cleaned.append({
                "head": h, "head_type": ht,
                "relation": rel,
                "tail": t, "tail_type": tt,
                "qualifiers": q
            })
        return cleaned

    # =========================
    # 🗄️ Upsert into Neo4j
    # =========================
    def _ensure_constraints(self, session):
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Brand) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Installment) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Metric) REQUIRE n.key IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:MetricValue) REQUIRE n.obs_id IS UNIQUE")

    def _merge_entity(self, session, name: str, t: str) -> str:
        label = _label_for(t)
        if t == "metric":
            key = _metric_canon(name) or name.lower().replace(" ", "_")
            session.run(f"MERGE (m:{_label_for('metric')} {{key:$key}}) ON CREATE SET m.display=$display",
                        key=key, display=name)
            return key
        canon = canonical_name_from_graph(name, label, session)
        session.run(f"MERGE (n:{label} {{name:$name}})", name=canon)
        return canon

    def _merge_metric_value(self, session, head_label: str, head_name: str, metric_key: str,
                            qualifiers: Dict[str, Any], source_document_id: Optional[str], chunk_id: Optional[str]):
        val = qualifiers.get("value")
        unit = qualifiers.get("unit")
        period = qualifiers.get("time_period") or qualifiers.get("date")
        segment = qualifiers.get("segment")
        population_segment = qualifiers.get("population_segment")
        region = qualifiers.get("region")
        platform = qualifiers.get("platform")

        obs_id = _hash_observation(head_name, metric_key, val, unit, period, segment, population_segment, region, platform)

        # Subject (head)
        session.run(f"MERGE (h:{head_label} {{name:$hname}})", hname=head_name)

        # Metric + observation
        session.run(f"""
            MERGE (m:Metric {{key:$metric_key}})
            ON CREATE SET m.display = $metric_key
            MERGE (mv:MetricValue {{obs_id:$obs_id}})
            ON CREATE SET mv.value=$val, mv.unit=$unit, mv.platform=$platform
            SET mv.value = COALESCE($val, mv.value),
                mv.unit = COALESCE($unit, mv.unit),
                mv.platform = COALESCE($platform, mv.platform)
            MERGE (h)-[:HAS_METRIC_VALUE]->(mv)
            MERGE (mv)-[:OF_METRIC]->(m)
        """, metric_key=metric_key, obs_id=obs_id, val=val, unit=unit, platform=platform)

        # Qualifier nodes
        if segment:
            session.run("MERGE (s:PlayerSegment {name:$name})", name=segment)
            session.run("""
                MATCH (mv:MetricValue {obs_id:$obs_id}), (s:PlayerSegment {name:$name})
                MERGE (mv)-[:TARGET_SEGMENT]->(s)
            """, obs_id=obs_id, name=segment)

        if population_segment:
            session.run("MERGE (ps:PlayerSegment {name:$name})", name=population_segment)
            session.run("""
                MATCH (mv:MetricValue {obs_id:$obs_id}), (ps:PlayerSegment {name:$name})
                MERGE (mv)-[:WITHIN_POPULATION]->(ps)
            """, obs_id=obs_id, name=population_segment)

        if region:
            session.run("MERGE (r:Region {name:$name})", name=region)
            session.run("""
                MATCH (mv:MetricValue {obs_id:$obs_id}), (r:Region {name:$name})
                MERGE (mv)-[:FOR_REGION]->(r)
            """, obs_id=obs_id, name=region)

        if period:
            session.run("MERGE (tp:TimePeriod {name:$name})", name=period)
            session.run("""
                MATCH (mv:MetricValue {obs_id:$obs_id}), (tp:TimePeriod {name:$name})
                MERGE (mv)-[:DURING_PERIOD]->(tp)
            """, obs_id=obs_id, name=period)

        if platform:
            session.run("MERGE (p:Platform {name:$name})", name=platform)
            session.run("""
                MATCH (mv:MetricValue {obs_id:$obs_id}), (p:Platform {name:$name})
                MERGE (mv)-[:ON_PLATFORM]->(p)
            """, obs_id=obs_id, name=platform)

        # Create proper relationship hierarchy: Document -> Chunk -> Entity -> MetricValue
        if source_document_id and chunk_id:
            # Link the metric value to the chunk, and chunk to document
            session.run("""
                MATCH (mv:MetricValue {obs_id:$obs_id}), (c:Chunk {chunk_id:$cid}), (d:Document {source_id:$sid})
                MERGE (mv)-[:MENTIONED_IN_CHUNK]->(c)
                MERGE (c)-[:BELONGS_TO_DOCUMENT]->(d)
            """, obs_id=obs_id, cid=chunk_id, sid=source_document_id)
        elif source_document_id:
            # If no chunk, link directly to document
            session.run("""
                MATCH (mv:MetricValue {obs_id:$obs_id}), (d:Document {source_id:$sid})
                MERGE (mv)-[:DOCUMENTED_IN]->(d)
            """, obs_id=obs_id, sid=source_document_id)

    def upsert(self, payload: Dict[str, Any], source_document_id: str = None, chunk_id: str = None):
        entities = payload.get("entities") or []
        relations = payload.get("relations") or []
        if not entities and not relations:
            return

        with self.driver.session() as sess:
            self._ensure_constraints(sess)

            # Merge entities
            name_cache: Dict[str, Dict[str, str]] = {}
            for e in entities:
                name = _canon(e["name"]); t = e["type"]
                if not name or not _valid_type(t): continue
                label = _label_for(t)
                canon = self._merge_entity(sess, name, t)
                name_cache.setdefault(t, {})[name.lower()] = canon

                # alias → IS_SYNONYM_OF
                for alias in e.get("aliases") or []:
                    alias_canon = _canon(alias)
                    if alias_canon and alias_canon.lower() != canon.lower():
                        sess.run(f"""
                            MERGE (a:{label} {{name:$alias}})
                            MERGE (b:{label} {{name:$canon}})
                            MERGE (a)-[:IS_SYNONYM_OF]->(b)
                        """, alias=alias_canon, canon=canon)

            # Merge relations
            for r in relations:
                h = _canon(r["head"]); ht = r["head_type"]
                t = _canon(r["tail"]); tt = r["tail_type"]
                rel = r["relation"]; q = r.get("qualifiers") or {}
                if not h or not t or not _valid_type(ht) or not _valid_type(tt) or not _valid_rel(rel):
                    continue

                # Canon via cache/graph
                hcanon = name_cache.get(ht, {}).get(h.lower()) or self._merge_entity(sess, h, ht)
                tcanon = name_cache.get(tt, {}).get(t.lower()) or self._merge_entity(sess, t, tt)
                H, T = _label_for(ht), _label_for(tt)

                # Metric observation path if metric + qualifiers present (and policy_ok)
                metric_key = None
                if tt == "metric":
                    metric_key = _metric_canon(tcanon or t) or _metric_canon(t)
                elif ht == "metric":
                    metric_key = _metric_canon(hcanon or h) or _metric_canon(h)

                has_any_context = any(k in q for k in ("time_period", "date", "segment", "population_segment", "region", "platform"))
                has_value_signal = ("value" in q) or has_any_context

                if metric_key and has_value_signal and self._metrics_policy_ok(rel, ht, tt, q, hcanon, tcanon):
                    subj_label, subj_name = (H, hcanon) if tt == "metric" else (T, tcanon)
                    self._merge_metric_value(sess, subj_label, subj_name, metric_key, q, source_document_id, chunk_id)
                    # Lightweight association for traversal
                    sess.run(f"""
                        MATCH (s:{subj_label} {{name:$sname}}), (m:Metric {{key:$mkey}})
                        MERGE (s)-[:HAS_METRIC]->(m)
                    """, sname=subj_name, mkey=metric_key)
                    continue

                # Generic relation upsert (w/ scoped qualifiers to nodes)
                cypher = f"""
                    MERGE (h:{H} {{name:$h}})
                    MERGE (t:{T} {{name:$t}})
                    MERGE (h)-[rel:{rel.upper()}]->(t)
                """
                params = {"h": hcanon, "t": tcanon}

                if "time_period" in q:
                    cypher += """
                        WITH h, t, rel
                        MERGE (tp:TimePeriod {name:$tp})
                        MERGE (h)-[:DURING_PERIOD]->(tp)
                    """
                    params["tp"] = q["time_period"]
                if "date" in q:
                    cypher += """
                        WITH h, t, rel
                        MERGE (d:Date {name:$date})
                        MERGE (h)-[:OCCURRED_ON]->(d)
                    """
                    params["date"] = q["date"]
                if "segment" in q:
                    cypher += """
                        WITH h, t, rel
                        MERGE (s:PlayerSegment {name:$seg})
                        MERGE (h)-[:FOR_SEGMENT]->(s)
                    """
                    params["seg"] = q["segment"]
                if "population_segment" in q:
                    cypher += """
                        WITH h, t, rel
                        MERGE (ps:PlayerSegment {name:$pseg})
                        MERGE (h)-[:WITHIN_POPULATION]->(ps)
                    """
                    params["pseg"] = q["population_segment"]
                if "region" in q:
                    cypher += """
                        WITH h, t, rel
                        MERGE (r:Region {name:$reg})
                        MERGE (h)-[:FOR_REGION]->(r)
                    """
                    params["reg"] = q["region"]
                if "platform" in q:
                    cypher += """
                        WITH h, t, rel
                        MERGE (p:Platform {name:$plat})
                        MERGE (h)-[:ON_PLATFORM]->(p)
                    """
                    params["plat"] = q["platform"]

                # Note: Primary entity linking is now handled separately in process_chunks
                # to implement hierarchical clustering based on relationship centrality

                try:
                    sess.run(cypher, **params)
                except Exception as e:
                    print(f"⚠️ Neo4j write failed for {hcanon}-{rel}->{tcanon}: {e}")
            
            self._ensure_brand_installment_connections(sess)
    
    def _identify_primary_entities(self, entities: List[Dict], relations: List[Dict]) -> List[str]:
        """
        Identify primary entities using hierarchical clustering based on relationship centrality.
        Returns entities that should be linked to chunks (not their children).
        """
        if not entities:
            return []
        
        # If no relations, make ALL entities primary to prevent orphans
        if not relations:
            print(f"    🔗 No relations found, making ALL entities primary: {[e['name'] for e in entities]}")
            return [e["name"] for e in entities]
        
        # Build entity name to entity mapping
        entity_map = {e["name"]: e for e in entities}
        entity_names = set(entity_map.keys())
        
        # Build relationship graph
        connections = {name: set() for name in entity_names}
        for rel in relations:
            head_name = rel["head"]
            tail_name = rel["tail"]
            if head_name in entity_names and tail_name in entity_names:
                connections[head_name].add(tail_name)
                connections[tail_name].add(head_name)
        
        # Hierarchical clustering: iteratively find most connected entities
        primary_entities = []
        remaining_entities = entity_names.copy()
        
        while remaining_entities:
            # Find entity with most connections among remaining entities
            max_connections = 0
            candidates = []
            
            for entity_name in remaining_entities:
                # Count connections to other remaining entities
                connected_count = len(connections[entity_name] & remaining_entities)
                if connected_count > max_connections:
                    max_connections = connected_count
                    candidates = [entity_name]
                elif connected_count == max_connections and connected_count > 0:
                    candidates.append(entity_name)
            
            if not candidates or max_connections == 0:
                # No more connected entities, handle remaining entities
                # CRITICAL: ALL remaining entities must be made primary to prevent orphans
                print(f"    🔗 No more connected entities, making ALL remaining primary: {remaining_entities}")
                primary_entities.extend(remaining_entities)
                remaining_entities.clear()
                break
            
            # Add all tied candidates as primary entities
            primary_entities.extend(candidates)
            
            # Remove primary entities and their direct connections
            to_remove = set(candidates)
            for candidate in candidates:
                to_remove.update(connections[candidate] & remaining_entities)
            
            remaining_entities -= to_remove
        
        # FINAL SAFETY CHECK: Ensure no entities are left unlinked
        all_entity_names = {e["name"] for e in entities}
        primary_set = set(primary_entities)
        unlinked_entities = all_entity_names - primary_set
        
        if unlinked_entities:
            print(f"    ⚠️ Found unlinked entities, adding as primary: {unlinked_entities}")
            primary_entities.extend(unlinked_entities)
        
        # CRITICAL: If no primary entities were found, make ALL entities primary
        if not primary_entities:
            print(f"    🚨 CRITICAL: No primary entities found, making ALL entities primary: {all_entity_names}")
            primary_entities = list(all_entity_names)
        
        print(f"    ✅ Final primary entities: {len(primary_entities)} out of {len(all_entity_names)} total")
        return primary_entities

    def _link_primary_entities_to_chunk(self, primary_entities: List[str], chunk_id: str, source_document_id: str = None):
        """Link primary entities to chunks and documents."""
        with self.driver.session() as sess:
            for entity_name in primary_entities:
                # Link entity to chunk
                sess.run("""
                    MATCH (e {name:$entity_name}), (c:Chunk {chunk_id:$chunk_id})
                    MERGE (e)-[:MENTIONED_IN_CHUNK]->(c)
                """, entity_name=entity_name, chunk_id=chunk_id)
                
                # Link chunk to document if source_document_id provided
                if source_document_id:
                    sess.run("""
                        MATCH (c:Chunk {chunk_id:$chunk_id}), (d:Document {source_id:$source_id})
                        MERGE (c)-[:BELONGS_TO_DOCUMENT]->(d)
                    """, chunk_id=chunk_id, source_id=source_document_id)

    # =========================
    # 🔗 Brand-Installment hygiene
    # =========================
    def _ensure_brand_installment_connections(self, session):
        result = session.run("""
            MATCH (i:Installment)
            WHERE NOT EXISTS( (i)-[:BELONGS_TO_BRAND]->() )
            RETURN i.name AS name
        """)
        for rec in result:
            name = rec["name"]
            brand = self._extract_brand_from_installment(name)
            if brand:
                session.run("""
                    MERGE (b:Brand {name:$brand})
                    MERGE (i:Installment {name:$inst})
                    MERGE (i)-[:BELONGS_TO_BRAND]->(b)
                """, brand=brand, inst=name)
                print(f"🔗 Connected {name} → {brand}")

    def _extract_brand_from_installment(self, installment_name: str) -> Optional[str]:
        low = installment_name.lower()
        if "assassin" in low or "creed" in low:
            return "Assassin's Creed"
        if "far cry" in low or re.search(r"\bfc\s*\d\b", low):
            return "Far Cry"
        if "rainbow six" in low or "r6" in low:
            return "Tom Clancy's Rainbow Six"
        return None
    
    # =========================
    # 📦 Orchestration
    # =========================
    def process_chunks(self, chunks: List, source_document_id: str = None):
        print(f"    📊 Processing {len(chunks)} chunks for knowledge extraction...")
        total_rel = 0
        total_ent = 0
        all_triples = []
        
        for i, ch in enumerate(chunks):
            txt = ch.page_content
            result = self.extract_structured(txt)  # use structured to enforce policies
            entities = result["entities"]
            relations = result["relations"]

            if entities or relations:
                chunk_id = ch.metadata.get("chunk_id") if hasattr(ch, 'metadata') else None
                print(f"      🔍 DEBUG: chunk_id = {chunk_id}")
                
                # First, upsert all entities and relations to build the graph
                self.upsert(result, source_document_id, chunk_id)
                
                # Then, identify primary entities using hierarchical clustering
                print(f"      🔍 DEBUG: Starting hierarchical clustering for {len(entities)} entities")
                primary_entities = self._identify_primary_entities(entities, relations)
                print(f"      🔍 DEBUG: Identified {len(primary_entities)} primary entities: {primary_entities}")
                
                # Link only primary entities to chunks
                if primary_entities and chunk_id:
                    print(f"      🔍 DEBUG: Linking {len(primary_entities)} entities to chunk {chunk_id}")
                    self._link_primary_entities_to_chunk(primary_entities, chunk_id, source_document_id)
                    print(f"      🔍 DEBUG: Linking completed")
                else:
                    print(f"      🔍 DEBUG: Skipping link - primary_entities: {len(primary_entities) if primary_entities else 0}, chunk_id: {chunk_id}")
                
                print(f"      • chunk {i+1}/{len(chunks)}: {len(entities)} entities, {len(relations)} relations, {len(primary_entities)} primary")
                total_ent += len(entities); total_rel += len(relations)
                for r in relations:
                    all_triples.append({
                        "head": r["head"], "head_type": r["head_type"],
                        "relation": r["relation"], "tail": r["tail"], "tail_type": r["tail_type"]
                    })
            else:
                print(f"      • chunk {i+1}/{len(chunks)}: no high-signal facts")
        
        print(f"    ✅ Total entities: {total_ent} | Total relations: {total_rel}")
        if all_triples:
            self._analyze_ontology_gaps(all_triples)
    
    def _analyze_ontology_gaps(self, triples: List[Dict[str, Any]]):
        try:
            expander = OntologyExpander()
            suggestions = expander.analyze_triple_gaps(triples)
            if suggestions:
                print(f"    🔍 Found {len(suggestions)} potential ontology gaps:")
                for s in suggestions:
                    print(f"      • {s['type']}: {s['original_type']} → {s['canonical_name']} (confidence: {s['confidence']:.2f})")
                    expander.add_suggestion(s)
                print("    📝 Suggestions saved to ontology_suggestions.json")
                print("    💡 Review/approve via ontology management UI")
            else:
                print("    ✅ No ontology gaps detected")
        except Exception as e:
            print(f"    ⚠️ Ontology gap analysis failed: {e}")
    
    def store_triple(self, chunk_id: str, triple: tuple, source_document_id: str = None):
        try:
            h, ht, r, t, tt = triple
            payload = {
                "entities": [{"name": h, "type": ht}, {"name": t, "type": tt}],
                "relations": [{"head": h, "head_type": ht, "relation": r, "tail": t, "tail_type": tt}]
            }
            self.upsert(payload, source_document_id, chunk_id)
            return True
        except Exception as e:
            print(f"❌ Error storing triple {triple}: {e}")
            return False

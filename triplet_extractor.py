import os
import ast
import json
from typing import List, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re

load_dotenv()

ONTOLOGY_PATH = "ontology.json"


def load_ontology():
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ontology(ontology):
    with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)


def safe_label(label):
    label = label.strip().title()
    label = re.sub(r"[\s\-]", "", label)  # Remove space and hyphen
    label = re.sub(r"\W", "", label)  # Remove all non-word chars
    return label


class TripletExtractor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Load ontology and store as class variables
        self.ontology = load_ontology()
        self.entity_types = self.ontology["entity_types"]
        self.relation_types = self.ontology["relation_types"]

        entity_types_str = ", ".join(self.entity_types)
        relation_types_str = ", ".join(self.relation_types)

        self.prompt = PromptTemplate.from_template(
            f"""
You are an expert in extracting structured information from Indian government documents.

Extract all factual relationships from the text below as [SUBJECT, SUBJECT_TYPE, RELATION, OBJECT, OBJECT_TYPE].

Only use the following allowed entity types: {entity_types_str}
Only use the following allowed relation types: {relation_types_str}

If a new entity or relation type is absolutely needed, invent it - while sticking to the format of existing entities and relations such that they can be generalized and are not too specific, flag it with NEW_ENTITY_TYPE or NEW_RELATION_TYPE, and output it.
(Example: ["X", "NEW_ENTITY_TYPE:mission", "has_objective", "Y", "organization"])

Example:
prompt: “The Ministry of Finance, headed by Nirmala Sitharaman, announced the Fiscal Responsibility Act on 1 Feb 2024. ₹1 lakh crore was allocated to the PMJDY scheme, which is implemented by the Department of Financial Services.”
output:
[
["Ministry of Finance", "ministry", "has_minister", "Nirmala Sitharaman", "person"],
["Nirmala Sitharaman", "person", "has_title", "Finance Minister", "title"],
["Ministry of Finance", "ministry", "announced", "Fiscal Responsibility Act", "policy"],
["Fiscal Responsibility Act", "policy", "start_date", "1 Feb 2024", "date"],
["PMJDY", "scheme", "implemented_by", "Department of Financial Services", "department"]
]

Now process this:
```{{text}}```
output:
            """
        )
        self.chain = self.prompt | self.llm

    def parse_triplets(self, raw_output: str):
        """Parse and return triplets, track any new types for ontology update."""
        cleaned = (
            raw_output.replace("```json", "")
            .replace("```", "")
            .replace("\n", "")
            .strip()
        )
        print("📥 Cleaned raw input to eval:\n", cleaned)
        try:
            triplet_list = ast.literal_eval(cleaned)
        except Exception as e:
            print("⚠️ Failed to parse structured triplets:", e)
            return [], set(), set()

        valid = []
        new_entities, new_relations = set(), set()
        for triplet in triplet_list:
            if isinstance(triplet, list) and len(triplet) == 5:
                s, s_type, p, o, o_type = (x.strip(" \"'[]") for x in triplet)
                if s_type.startswith("NEW_ENTITY_TYPE:"):
                    ent_type = s_type.split(":", 1)[1].strip().lower()
                    new_entities.add(ent_type)
                    s_type = ent_type
                if o_type.startswith("NEW_ENTITY_TYPE:"):
                    ent_type = o_type.split(":", 1)[1].strip().lower()
                    new_entities.add(ent_type)
                    o_type = ent_type
                if p.startswith("NEW_RELATION_TYPE:"):
                    rel_type = p.split(":", 1)[1].strip().lower()
                    new_relations.add(rel_type)
                    p = rel_type
                valid.append((s, s_type, p, o, o_type))
        return valid, new_entities, new_relations

    def update_ontology(self, new_entity_types, new_relation_types):
        updated = False
        for ent in new_entity_types:
            if ent not in self.entity_types:
                print(f"🆕 Adding new entity type to ontology: {ent}")
                self.entity_types.append(ent)
                updated = True
        for rel in new_relation_types:
            if rel not in self.relation_types:
                print(f"🆕 Adding new relation type to ontology: {rel}")
                self.relation_types.append(rel)
                updated = True
        if updated:
            self.ontology["entity_types"] = self.entity_types
            self.ontology["relation_types"] = self.relation_types
            save_ontology(self.ontology)

    def insert_into_neo4j(self, triplets, chunk_id):
        with self.driver.session() as session:
            for s, s_type, p, o, o_type in triplets:
                s_label = safe_label(s_type)
                o_label = safe_label(o_type)
                rel_type = p.strip().upper().replace(" ", "_").replace("-", "_")
                # Create or update the entity nodes and their relation
                query = f"""
                MERGE (a:{s_label} {{name: $s}})
                MERGE (b:{o_label} {{name: $o}})
                MERGE (a)-[r:{rel_type}]->(b)
                """
                session.run(query, s=s.strip(), o=o.strip())
                # Also link the chunk to both subject and object entities
                mention_query = f"""
                MERGE (c:Chunk {{chunk_id: $chunk_id}})
                MERGE (a:{s_label} {{name: $s}})
                MERGE (b:{o_label} {{name: $o}})
                MERGE (c)-[:MENTIONS]->(a)
                MERGE (c)-[:MENTIONS]->(b)
                """
                session.run(mention_query, chunk_id=chunk_id, s=s.strip(), o=o.strip())

    def process_chunks(self, chunks):
        for i, chunk in enumerate(chunks):
            print(f"\n📄 Processing chunk {i + 1}/{len(chunks)}")
            try:
                response = self.chain.invoke({"text": chunk.page_content})
                print("📥 Raw triplet output:\n", response.content)
                triplets, new_ents, new_rels = self.parse_triplets(response.content)
                if new_ents or new_rels:
                    self.update_ontology(new_ents, new_rels)
                if triplets:
                    # Robustly extract the chunk_id from chunk (prefer metadata)
                    chunk_id = None
                    # Try attribute first
                    if hasattr(chunk, "chunk_id"):
                        chunk_id = chunk.chunk_id
                    # Then try metadata
                    if not chunk_id and hasattr(chunk, "metadata"):
                        chunk_id = chunk.metadata.get("chunk_id")
                    # Fallback: use index
                    if not chunk_id:
                        chunk_id = f"unknown_chunk_{i}"
                    self.insert_into_neo4j(triplets, chunk_id)
                    print(
                        f"✅ Inserted {len(triplets)} triplets into Neo4j (chunk_id: {chunk_id})"
                    )
                else:
                    print("⚠️ No valid triplets found.")
            except Exception as e:
                print(f"❌ Error processing chunk {i + 1}: {e}")

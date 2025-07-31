# nodes/compose_context.py
from query_state import QueryState


def compose_context(state: QueryState) -> QueryState:
    sections = []
    chunks = state["chunks"]
    mentions = state["mentions"]
    triples = state["expansion_triples"]

    for i, chunk in enumerate(chunks):
        section = f"Chunk {i+1}:\n{chunk}\nMentions:"
        for name, label in mentions:
            section += f"\n- {name} ({label})"
        sections.append(section)

    if triples:
        sections.append("Expanded facts:\n" + "\n".join(triples))

    state["context"] = "\n\n---\n\n".join(sections)
    return state

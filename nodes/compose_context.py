# nodes/compose_context.py
from query_state import QueryState


def compose_context(state: QueryState) -> QueryState:
    chunks = state.get("chunks", [])
    mentions = state.get("mentions", [])
    triples = state.get("expansion_triples", [])

    sections = []

    for i, chunk in enumerate(chunks):
        section = f"Chunk {i+1}:\n{chunk}\nMentions:"
        found_mentions = [
            f"- {name} ({label})" for name, label in mentions if name in chunk
        ]
        if found_mentions:
            section += "\n" + "\n".join(found_mentions)
        else:
            section += "\n- (none)"
        sections.append(section)

    if triples:
        sections.append("Expanded facts:\n" + "\n".join(triples))

    context = "\n\n---\n\n".join(sections)
    print("🧱 Context built:\n", context[:500])  # optional debug

    state["context"] = context

    return state

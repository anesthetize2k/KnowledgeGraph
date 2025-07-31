# run_query.py
from query_graph import graph

print("🔁 Recompiling graph...")
app = graph.compile()

print("\n🧠 KnowledgeGraph Query Engine")
print("Type your question (or type 'exit' to quit)\n")

while True:
    question = input("💬 ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    if not question:
        continue

    try:
        result = app.invoke({"question": question})
        print("\n🤖", result["final_answer"])
    except Exception as e:
        print(f"❌ Error: {e}")

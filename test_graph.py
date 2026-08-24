"""
test_graph.py

Runs the FULL LangGraph pipeline end-to-end -- router + all three agents
wired together, exactly as the real app will use them.

Now that the graph has a checkpointer, each thread_id automatically
remembers its own state between calls. Notice scenario 3b: we no longer
manually pass pending_question -- the checkpointer already knows it,
as long as we reuse the SAME thread_id as 3a.

Three scenarios:
  1. A question answerable from your notes -> routed to Learning agent
  2. A question NOT in your notes -> Learning agent flags needs_research,
     and the graph AUTOMATICALLY continues to the Research agent in the
     same call
  3. A quiz request, followed by a simulated answer on the SAME thread ->
     turn 2 skips the router's LLM call entirely, since the checkpointer
     already has pending_question saved from turn 1
"""

from workflow import app_graph


def run(query, current_topic="", thread_id="test-thread"):
    config = {"configurable": {"thread_id": thread_id}}
    return app_graph.invoke(
        {"query": query, "current_topic": current_topic},
        config=config,
    )


print("=== 1. Question IN your notes ===")
r1 = run("What is a deterministic finite automaton?", thread_id="t1")
print(f"Routed to: {r1['router_decision']}")
print(r1["learning_agent_output"])

print("\n=== 2. Question NOT in your notes (auto fallback to research) ===")
r2 = run("What is the time complexity of quicksort in the worst case?", thread_id="t2")
print(f"Routed to: {r2['router_decision']}, needs_research: {r2['needs_research']}")
print(r2["research_agent_output"] or r2["learning_agent_output"])

print("\n=== 3a. Quiz request ===")
r3 = run("Quiz me on DFAs", current_topic="Deterministic Finite Automata", thread_id="t3")
print(f"Routed to: {r3['router_decision']}")
print(r3["quiz_agent_output"])

print("\n=== 3b. Answering the quiz question (same thread_id -- no manual state passing) ===")
r4 = run(
    "A DFA has a finite set of states, an alphabet, a transition function, a start state, and accepting states.",
    thread_id="t3",  # same thread as 3a -- that's the only link needed now
)
print(f"Routed to: {r4['router_decision']} (should be 'quiz' -- checkpointer remembered pending_question)")
print(r4["quiz_agent_output"])

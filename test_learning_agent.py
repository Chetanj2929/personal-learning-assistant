"""
test_learning_agent.py

Tests the LearningAgent directly (no graph yet — that's the next step).
Assumes you've already run test_rag.py once, so your notes are ingested
into data_vector_db/.

Runs two questions on purpose:
  1. One that SHOULD be answerable from sample_note.txt
  2. One that should NOT be — to see the fallback signal fire correctly
"""

from agents.learning_agent import LearningAgent

agent = LearningAgent()

print("=== Question covered by your notes ===")
state = {
    "query": "What is a deterministic finite automaton?",
    "current_topic": "TOC",
}
result = agent.run(state)
print(result["learning_agent_output"])
print(f"\nneeds_research: {result['needs_research']}\n")

print("=== Question NOT covered by your notes ===")
state2 = {
    "query": "What is the time complexity of quicksort in the worst case?",
    "current_topic": "DAA",
}
result2 = agent.run(state2)
print(result2["learning_agent_output"])
print(f"\nneeds_research: {result2['needs_research']}")

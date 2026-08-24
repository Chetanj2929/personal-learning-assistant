"""
test_quiz_agent.py

Simulates a two-turn quiz conversation:
  1. Ask for a quiz question (no pending_question yet -> generates one)
  2. "Answer" it (pending_question now set -> grades instead of generating)

Assumes test_rag.py has already been run, so your notes are ingested.
"""

from agents.quiz_agent import QuizAgent

agent = QuizAgent()

# --- Turn 1: generate a question ---
state = {
    "query": "quiz me",
    "current_topic": "Deterministic Finite Automata",
    "pending_question": None,
}
state = agent.run(state)

print("=== Quiz question ===")
print(state["quiz_agent_output"])
print(f"\npending_question is now set: {bool(state['pending_question'])}\n")

# --- Turn 2: "answer" it ---
state["query"] = (
    "A DFA is a finite state machine where each state has exactly one "
    "transition for every input symbol, so there's no ambiguity about "
    "which state comes next."
)
state = agent.run(state)

print("=== Grading feedback ===")
print(state["quiz_agent_output"])
print(f"\npending_question cleared: {state['pending_question'] is None}")

"""
state.py

This is the shared "whiteboard" that flows through every node in the graph.
Each node (router, learning agent, quiz agent, research agent) reads what's
already on it, does its job, and writes updates back before LangGraph passes
it on to the next node.

Everything is Optional except the fields we know will always be set by the
time the graph starts running.
"""

from typing import TypedDict, Optional


class LearningState(TypedDict):
    # --- What the user just sent ---
    query: str                          # raw text from the chat input
    current_topic: str                  # topic set in the sidebar, e.g. "Python OOP"

    # --- Quiz bookkeeping ---
    # When the quiz agent asks a question, it stores it here. On the user's
    # NEXT message, the router sees this is set and sends them straight back
    # to the quiz agent to be graded, instead of re-routing their answer.
    pending_question: Optional[str]
    last_quiz_result: Optional[str]     # e.g. "correct", "incorrect: ..."

    # --- Routing ---
    router_decision: Optional[str]      # "learning" | "quiz" | "research"
    needs_research: Optional[bool]       # set True if the learning agent found nothing in your notes

    # --- Each agent's output ---
    # Kept as separate fields (rather than overwriting one "answer" field)
    # so you can log/debug which agent produced what.
    learning_agent_output: Optional[str]
    quiz_agent_output: Optional[str]
    research_agent_output: Optional[str]

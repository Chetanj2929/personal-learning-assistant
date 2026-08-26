"""
workflow.py

This is where LangGraph actually wires the pieces together into a graph:

        router
       /   |   \\
 learning quiz research
     |
  needs_research?
   /        \\
 Yes         No
  |           |
research     END

The router only calls the LLM to classify when there's a genuinely NEW
question (no pending_question). If you're mid-quiz, we already KNOW
where the message goes — no wasted LLM call guessing something we
already know from state.
"""

import os
import re

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from state import LearningState
from agents.learning_agent import LearningAgent
from agents.quiz_agent import QuizAgent
from agents.research_agent import ResearchAgent

load_dotenv()

_router_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,  # deterministic — this is classification, not creativity
)

ROUTER_PROMPT = """Classify the student's message into exactly ONE of these
three categories. Respond with ONLY the single word, nothing else.

- learning: they're asking a question and want an explanation
- quiz: they're asking to be quizzed or tested on a topic
- research: they're explicitly asking you to look something up online

Message: {query}"""

# Agents are instantiated once and reused across every request — each one
# loads its own LLM client and (for learning/quiz) the RAG service.
_learning_agent = LearningAgent()
_quiz_agent = QuizAgent()
_research_agent = ResearchAgent()


def router_node(state: LearningState) -> LearningState:
    if state.get("pending_question"):
        state["router_decision"] = "quiz"
        return state

    prompt = ROUTER_PROMPT.format(query=state["query"])
    response = _router_llm.invoke([HumanMessage(content=prompt)])
    decision = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip().lower()

    if decision not in ("learning", "quiz", "research"):
        decision = "learning"  # safe default if the LLM answers oddly

    state["router_decision"] = decision
    return state


def learning_node(state: LearningState) -> LearningState:
    return _learning_agent.run(state)


def quiz_node(state: LearningState) -> LearningState:
    return _quiz_agent.run(state)


def research_node(state: LearningState) -> LearningState:
    return _research_agent.run(state)


def route_after_router(state: LearningState) -> str:
    return state["router_decision"]


def route_after_learning(state: LearningState) -> str:
    return "research" if state.get("needs_research") else "end"


def build_graph():
    graph = StateGraph(LearningState)

    graph.add_node("router", router_node)
    graph.add_node("learning", learning_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("research", research_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"learning": "learning", "quiz": "quiz", "research": "research"},
    )

    # The key fallback wiring: after the learning agent runs, check its
    # own needs_research flag and continue to research automatically —
    # all within a SINGLE graph.invoke() call, no extra code needed.
    graph.add_conditional_edges(
        "learning",
        route_after_learning,
        {"research": "research", "end": END},
    )

    graph.add_edge("quiz", END)
    graph.add_edge("research", END)

    # A checkpointer means: calling app_graph.invoke(..., config={"configurable":
    # {"thread_id": "some-id"}}) automatically remembers state (pending_question,
    # current_topic, etc.) from the LAST call with that same thread_id. You only
    # need to send what's NEW each turn -- the rest persists on its own.
    return graph.compile(checkpointer=InMemorySaver())


# Compiled once at import time, reused across every request.
app_graph = build_graph()

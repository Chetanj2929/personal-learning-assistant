"""
agents/research_agent.py

The "fallback." Only runs when the Learning agent's needs_research flag
came back True — meaning your notes didn't have the answer. This agent
searches the web instead (via ddgs, free, no API key), then asks the LLM
to synthesize a clear answer from the results, clearly labeled as coming
from the web rather than your own notes — so you always know the source.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS

load_dotenv()

SYSTEM_PROMPT = """You are a helpful research assistant. Using ONLY the
search results below, answer the student's question clearly and concisely
— like you're explaining it to a classmate, not writing a report. If the
results don't actually answer the question, say so honestly instead of
guessing."""


class ResearchAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="qwen/qwen3.6-27b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

    def _web_search(self, query: str, max_results: int = 5):
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    def run(self, state: dict) -> dict:
        query = state["query"]

        try:
            results = self._web_search(query)
        except Exception as e:
            state["research_agent_output"] = (
                f"Web search failed ({e}). Check your internet connection and try again."
            )
            return state

        if not results:
            state["research_agent_output"] = (
                "I searched the web but couldn't find anything useful for that question."
            )
            return state

        context = "\n\n".join(
            f"[{r.get('title', 'Untitled')}]\n{r.get('body', '')}"
            for r in results
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Search results:\n{context}\n\nQuestion: {query}"),
        ]
        response = self.llm.invoke(messages)
        answer = response.content.strip()

        state["research_agent_output"] = (
            f"{answer}\n\n_(This wasn't in your notes — found via web search.)_"
        )
        return state

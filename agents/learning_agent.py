"""
agents/learning_agent.py

The "explainer." Given a question, it:
  1. Retrieves relevant chunks from your notes via RAGService
  2. Asks the LLM to answer USING ONLY those chunks (grounded generation —
     the model can't just make something up from what it memorized)
  3. If the notes don't contain the answer, it says so explicitly — which
     is the signal the graph will later use to fall back to the Research
     agent instead of leaving you with a wrong or hallucinated answer.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from services.rag_service import RAGService

load_dotenv()  # reads GROQ_API_KEY from your .env file

# A marker the LLM outputs when it genuinely can't answer from context.
# We check for this EXACT string, so the prompt below is strict about it.
NOT_FOUND_MARKER = "NOT_FOUND_IN_NOTES"

SYSTEM_PROMPT = f"""You are a study assistant helping a student understand their own notes.

Answer the student's question using ONLY the context provided below — do not
use outside knowledge, even if you happen to know the answer some other way.

If the context does not contain enough information to answer the question,
respond with EXACTLY this and nothing else: {NOT_FOUND_MARKER}

Otherwise, be clear and concise, like you're explaining it to a classmate,
not writing a textbook chapter."""


class LearningAgent:
    def __init__(self):
        self.rag = RAGService()
        self.llm = ChatGroq(
            model="qwen/qwen3.6-27b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

    def run(self, state: dict) -> dict:
        query = state["query"]
        chunks = self.rag.retrieve(query, k=4)

        if not chunks:
            # Nothing relevant in the vector store — don't even bother
            # asking the LLM, just flag for research.
            state["needs_research"] = True
            state["learning_agent_output"] = None
            return state

        context = "\n\n".join(
            f"[From {c.metadata.get('source', 'your notes')}]\n{c.page_content}"
            for c in chunks
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        if answer == NOT_FOUND_MARKER:
            state["needs_research"] = True
            state["learning_agent_output"] = None
        else:
            state["needs_research"] = False
            state["learning_agent_output"] = answer

        return state

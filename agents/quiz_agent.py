"""
agents/quiz_agent.py

The "examiner." Works in two phases, tracked by state['pending_question']:

  PHASE 1 (no pending question yet):
    Generate ONE quiz question from your notes on the current topic,
    save it to state['pending_question'], and show it to you.

  PHASE 2 (pending question exists):
    Your next message is treated as your ANSWER to that question, not a
    new question. The agent retrieves context again, grades your answer
    against it, gives feedback, then clears pending_question so the next
    message starts fresh.
"""

import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from services.rag_service import RAGService

load_dotenv()

GENERATE_PROMPT = """You are a quiz master helping a student revise for exams.

Using ONLY the context below, write exactly ONE clear quiz question about
{topic}. It can be short-answer or conceptual, not multiple choice.
Output ONLY the question text — no "Question:" prefix, no answer, no
explanation."""

GRADE_PROMPT = """You are grading a student's quiz answer.

Context (their notes):
{context}

Question asked: {question}
Student's answer: {answer}

Judge whether the answer is correct based on the context. Respond in
exactly this format:

Verdict: <Correct / Partially correct / Incorrect>
Feedback: <one or two sentences, encouraging but honest, filling in
whatever they missed>"""


class QuizAgent:
    def __init__(self):
        self.rag = RAGService()
        self.llm = ChatGroq(
            model="qwen/qwen3.6-27b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5,  # a bit more variety when generating questions
        )

    def run(self, state: dict) -> dict:
        if state.get("pending_question"):
            return self._grade(state)
        return self._generate(state)

    def _generate(self, state: dict) -> dict:
        topic = state.get("current_topic") or state["query"]
        chunks = self.rag.retrieve(topic, k=4)

        if not chunks:
            state["quiz_agent_output"] = (
                "I don't have any notes on that topic yet — add some to "
                "the data/ folder and re-ingest before quizzing you on it."
            )
            state["pending_question"] = None
            return state

        context = "\n\n".join(c.page_content for c in chunks)
        prompt = GENERATE_PROMPT.format(topic=topic)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Context:\n{context}"),
        ]
        response = self.llm.invoke(messages)
        question = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

        state["pending_question"] = question
        state["quiz_agent_output"] = question
        return state

    def _grade(self, state: dict) -> dict:
        question = state["pending_question"]
        answer = state["query"]  # this message IS the answer to `question`

        chunks = self.rag.retrieve(question, k=4)
        context = "\n\n".join(c.page_content for c in chunks)

        prompt = GRADE_PROMPT.format(context=context, question=question, answer=answer)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        feedback = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

        state["quiz_agent_output"] = feedback
        state["last_quiz_result"] = feedback
        state["pending_question"] = None  # clear it — next message starts fresh
        return state

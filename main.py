"""
main.py

FastAPI backend. Exposes one endpoint, POST /chat, that the Streamlit
frontend (built next) calls. Each request only sends the NEW message and
the topic -- the checkpointer we added in workflow.py remembers
everything else (pending_question, etc.) between calls, keyed by
thread_id. Since this is a personal single-student app, we default to
one ongoing thread_id ("default") -- no login system needed.

Run with:  uvicorn main:app --reload
Then open: http://127.0.0.1:8000/docs
for FastAPI's auto-generated interactive test page -- you can try /chat
right from the browser without writing a separate test script.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from workflow import app_graph
from services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, make sure the vector store isn't empty. This matters on
    # hosts with EPHEMERAL storage (e.g. Render's free tier) where every
    # fresh deploy starts with nothing on disk -- without this, the app
    # would come up with no notes to search until someone manually re-ran
    # ingestion.
    rag = RAGService()
    if rag.count() == 0:
        num_chunks = rag.ingest_folder()
        print(f"Startup: vector store was empty, ingested {num_chunks} chunks from data/")
    else:
        print(f"Startup: vector store already has {rag.count()} chunks, skipping ingest")
    yield


app = FastAPI(title="Personal Learning Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    current_topic: str = ""
    thread_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    source: str  # "learning" | "quiz" | "research"


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    result = app_graph.invoke(
        {"query": request.query, "current_topic": request.current_topic},
        config=config,
    )

    decision = result.get("router_decision", "learning")
    if decision == "quiz":
        answer = result.get("quiz_agent_output") or ""
    elif result.get("needs_research"):
        answer = result.get("research_agent_output") or ""
    else:
        answer = (
            result.get("learning_agent_output")
            or result.get("research_agent_output")
            or ""
        )

    return ChatResponse(answer=answer, source=decision)


@app.get("/")
def root():
    return {"status": "ok", "message": "Learning assistant API is running"}

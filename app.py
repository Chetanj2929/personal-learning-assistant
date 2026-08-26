"""
app.py

The Streamlit chat UI. Calls the FastAPI backend at API_URL.

In Docker (HF Spaces): FastAPI runs on localhost:8000 in the same container;
Streamlit calls it via HTTP. HF Spaces exposes port 7860.

LOCAL (two terminals):
    Terminal 1:  uvicorn main:app --reload
    Terminal 2:  streamlit run app.py

DEPLOYED (HF Spaces Docker):
    Both processes start via start.sh inside one container.
    API_URL defaults to http://localhost:8000/chat — no secrets needed.
"""

import os

import requests
import streamlit as st


def get_config(key: str, default: str) -> str:
    """Reads from Streamlit secrets first, then env vars, then default."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


API_URL = get_config("API_URL", "http://localhost:8000/chat")

# ---------------------------------------------------------------------------
# Page config & global styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Learning Assistant", page_icon="📚", layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #131a3d 100%);
    }
    h1 {
        background: linear-gradient(90deg, #7c93ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    section[data-testid="stSidebar"] {
        background: #0d1230;
        border-right: 1px solid #262c52;
    }
    [data-testid="stChatMessage"] {
        border-radius: 14px;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-top: 4px;
    }
    .source-learning { background: #12293f; color: #7dd3fc; }
    .source-quiz { background: #3a2a06; color: #fbbf24; }
    .source-research { background: #2c1a4d; color: #c4b5fd; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>📚 Personal Learning Assistant</h1>", unsafe_allow_html=True)
st.caption("Ask questions from your notes, get quizzed, or reach beyond them when needed.")

SOURCE_LABELS = {
    "learning": ("📖 From your notes", "source-learning"),
    "quiz": ("📝 Quiz", "source-quiz"),
    "research": ("🌐 Web research", "source-research"),
}

# ---------------------------------------------------------------------------
# Sidebar: topic + reset
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    current_topic = st.text_input(
        "Current topic",
        value=st.session_state.get("current_topic", ""),
        placeholder="e.g. Theory of Computation",
    )
    st.session_state["current_topic"] = current_topic

    st.caption(
        "Set a topic before saying 'quiz me' -- the quiz agent uses this "
        "to decide what to ask about."
    )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source"):
            label, css_class = SOURCE_LABELS.get(msg["source"], (msg["source"], ""))
            st.markdown(
                f'<span class="source-badge {css_class}">{label}</span>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# New message — POST to the FastAPI backend (localhost:8000 in Docker)
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask a question, or say 'quiz me'...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    source = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": user_input,
                        "current_topic": current_topic,
                        "thread_id": "default",
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                source = data["source"]
            except requests.exceptions.ConnectionError:
                answer = (
                    "⚠️ Can't reach the backend. In Docker mode this shouldn't "
                    "happen — check container logs. Locally, make sure "
                    "`uvicorn main:app --reload` is running."
                )
            except Exception as exc:
                answer = f"⚠️ Something went wrong: {exc}"

        st.markdown(answer)
        if source:
            label, css_class = SOURCE_LABELS.get(source, (source, ""))
            st.markdown(
                f'<span class="source-badge {css_class}">{label}</span>',
                unsafe_allow_html=True,
            )

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "source": source}
    )

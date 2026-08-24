"""
test_setup.py

Nothing to do with the agent yet — this just confirms your virtual
environment, installed packages, and Groq API key all work together
BEFORE we build anything on top of them.

Run it once. If you see a sentence printed back, you're good to go.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()  # reads GROQ_API_KEY from your .env file

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

response = llm.invoke([HumanMessage(content="Say hello in one short sentence.")])
print(response.content)

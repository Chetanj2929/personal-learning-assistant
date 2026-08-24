"""
test_research_agent.py

Tests the Research agent on its own — the same quicksort question that
earlier triggered needs_research: True in the Learning agent test.
This one actually goes and finds an answer on the web.
"""

from agents.research_agent import ResearchAgent

agent = ResearchAgent()

state = {"query": "What is the time complexity of quicksort in the worst case?"}
result = agent.run(state)

print(result["research_agent_output"])

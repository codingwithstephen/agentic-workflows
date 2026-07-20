"""Verifies DirectPromptAgent: a plain prompt→completion round trip."""
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import DirectPromptAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)
print(direct_agent_response)

# Where did this answer come from? Nothing was injected — no system prompt,
# no documents, no tools — so the only possible source is the parametric
# knowledge baked into gpt-3.5-turbo during its training.
print(
    "\nKnowledge source: the answer comes entirely from gpt-3.5-turbo's own "
    "training data (parametric knowledge); the agent added no external context."
)

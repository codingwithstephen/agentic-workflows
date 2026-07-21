"""Verifies AugmentedPromptAgent: same question, but through a persona."""
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)
print(augmented_agent_response)

print(
    "\nKnowledge source: still purely the LLM's parametric knowledge — the "
    "persona changes how the model speaks, not what it knows."
)
print(
    "Persona effect: the system prompt reframes the model as a professor, so "
    "the reply opens with 'Dear students,' and adopts a lecturing register "
    "instead of the terse one-liner the direct agent produces for the same "
    "question."
)

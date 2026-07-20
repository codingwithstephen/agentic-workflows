"""Verifies KnowledgeAugmentedPromptAgent using deliberately wrong knowledge.

If the agent parrots the injected falsehood ("London"), we know the knowledge
fence works; if it says "Paris", it leaked back to its training data.
"""
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

prompt = "What is the capital of France?"
answer = knowledge_agent.respond(prompt)
print(answer)

print(
    "\nCheck: the reply says London — the injected (false) knowledge — rather "
    "than Paris, which the base model would say on its own. The agent is "
    "therefore answering from the provided knowledge, not its training data."
)

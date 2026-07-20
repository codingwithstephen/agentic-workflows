"""Verifies RoutingAgent: three specialists, three prompts, and the router
must send each prompt to the right one on semantics alone ("Rome, Texas" vs
"Rome, Italy" is the interesting case)."""
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona, knowledge="You know everything about Texas"
)
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona, knowledge="You know everything about Europe"
)
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    "You are a college math professor",
    knowledge=(
        "You know everything about math. You take prompts with numbers, "
        "extract math formulas, and show the answer without explanation"
    ),
)

routing_agent = RoutingAgent(openai_api_key)
routing_agent.agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda q: texas_agent.respond(q),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda q: europe_agent.respond(q),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula and its result",
        "func": lambda q: math_agent.respond(q),
    },
]

test_prompts = (
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories",
)

for test_prompt in test_prompts:
    print(f"\n=== Prompt: {test_prompt} ===")
    print(routing_agent.route(test_prompt))

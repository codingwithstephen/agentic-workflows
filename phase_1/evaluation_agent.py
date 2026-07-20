"""Verifies EvaluationAgent: a critique loop wrapped around a worker.

The worker is seeded with wrong knowledge and a chatty persona; the evaluator
demands a bare city name, forcing at least one correction round.
"""
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

# Worker: professor persona + intentionally wrong fact, per the spec
persona_worker = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
worker_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_worker, knowledge)

# Evaluator: accepts only a bare city name
persona_evaluator = "an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."

evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_evaluator,
    evaluation_criteria,
    worker_agent,           # agent_to_evaluate
    10,                     # max_interactions
)

outcome = evaluation_agent.evaluate(prompt)

print("\n=== Final Result ===")
print(f"Final response: {outcome['final_response']}")
print(f"Evaluation: {outcome['evaluation']}")
print(f"Iterations: {outcome['iterations']}")

"""Verifies ActionPlanningAgent: a vague breakfast wish should come back as a
clean, ordered list of concrete steps pulled from the agent's knowledge."""
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import ActionPlanningAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

knowledge = """
Steps to make scrambled eggs:
1. Crack eggs into a bowl
2. Beat eggs with a fork until mixed
3. Heat a pan with butter or oil over medium heat
4. Pour the beaten eggs into the pan
5. Stir gently as the eggs cook
6. Remove from heat when eggs are just set
7. Serve immediately
"""

action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge)

prompt = "One morning I wanted to have scrambled eggs"
extracted = action_planning_agent.extract_steps_from_prompt(prompt)

print("Extracted action steps:")
for n, step in enumerate(extracted, start=1):
    print(f"  {n}. {step}")

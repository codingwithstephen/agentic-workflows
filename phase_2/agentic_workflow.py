# agentic_workflow.py — Phase 2
#
# A TPM-style prompt goes in; an action planner breaks it into steps, a
# router hands each step to the right role (PM / Program Manager / Dev
# Engineer), and every answer has to get past an evaluation loop before
# it counts.

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

SPEC_FILENAME = "Product-Spec-Email-Router.txt"

# ---------------------------------------------------------------------------
# Knowledge, personas, and evaluation criteria
# ---------------------------------------------------------------------------

knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Features are defined by grouping related user stories. "
    "Features are defined by giving them a descriptive name, providing a "
    "short description, listing the affected personas, and defining the "
    "user stories they group, in this format: Feature Name, Description, Key Functionality, User Benefit. "
    "Tasks are defined for each story and feature by identifying "
    "the required technical work items in this format: Task ID, Task Title, "
    "Related User Story, Description, Acceptance Criteria, Estimated Effort, Dependencies. "
    "A development Plan for a product contains all these components"
)

persona_product_manager = (
    "You are a Product Manager, you are responsible for defining the user stories for a product."
)
persona_program_manager = (
    "You are a Program Manager, you are responsible for defining the features for a product."
)
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups."
)
persona_dev_engineer = (
    "You are a Development Engineer, you are responsible for defining the development tasks for a product."
)
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story."
)

persona_evaluator = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier  for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)


def build_workflow(openai_api_key, product_spec):
    """Wire up every agent and return (action_planning_agent, routing_agent)."""

    action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

    # The PM is the only role that needs the actual spec in its knowledge
    knowledge_product_manager = (
        "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
        "The sentences always start with: As a "
        "Write several stories for the product spec below, where the personas are the different users of the product. "
        "The product spec is: " + product_spec
    )

    roles = {
        "Product Manager": (
            persona_product_manager,
            knowledge_product_manager,
            evaluation_criteria_product_manager,
        ),
        "Program Manager": (
            persona_program_manager,
            knowledge_program_manager,
            evaluation_criteria_program_manager,
        ),
        "Development Engineer": (
            persona_dev_engineer,
            knowledge_dev_engineer,
            evaluation_criteria_dev_engineer,
        ),
    }

    # Same deal for every role: draft an answer, don't return it until it
    # survives the evaluator.
    def make_support_function(persona, knowledge, criteria):
        knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)
        evaluation_agent = EvaluationAgent(
            openai_api_key,
            persona_evaluator,
            criteria,
            agent_to_evaluate=knowledge_agent,
            max_interactions=10,
        )

        def support_function(query):
            draft = knowledge_agent.respond(query)
            gated = evaluation_agent.evaluate(draft)
            return gated["final_response"]

        return support_function

    support_functions = {
        name: make_support_function(*spec) for name, spec in roles.items()
    }

    # The descriptions are deliberately exclusionary ("only", "does not") —
    # it keeps the three roles far apart in embedding space.
    routing_agent = RoutingAgent(openai_api_key)
    routing_agent.agents = [
        {
            "name": "Product Manager",
            "description": (
                "Responsible for defining product personas and user stories only. "
                "Does not define features or tasks. Does not group stories."
            ),
            "func": support_functions["Product Manager"],
        },
        {
            "name": "Program Manager",
            "description": (
                "Responsible for defining product features by grouping related "
                "user stories only. Does not define user stories or tasks."
            ),
            "func": support_functions["Program Manager"],
        },
        {
            "name": "Development Engineer",
            "description": (
                "Responsible for defining development tasks needed to implement "
                "user stories and features only. Does not define user stories or features."
            ),
            "func": support_functions["Development Engineer"],
        },
    ]

    return action_planning_agent, routing_agent


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    with open(SPEC_FILENAME, encoding="utf-8") as spec_file:
        product_spec = spec_file.read()

    action_planning_agent, routing_agent = build_workflow(openai_api_key, product_spec)

    print("\n*** Workflow execution started ***\n")

    # A golden prompt — the kind of thing a TPM would actually ask
    workflow_prompt = "What would the development tasks for this product be?"
    print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

    print("\nDefining workflow steps from the workflow prompt")
    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
    print(f"Workflow steps: {workflow_steps}")

    completed_steps = []

    for step_number, step in enumerate(workflow_steps, start=1):
        print(f"\n=== Step {step_number}/{len(workflow_steps)}: {step} ===")
        step_result = routing_agent.route(step)
        completed_steps.append(step_result)
        print(f"Step result:\n{step_result}")

    print("\n*** Workflow execution completed ***\n")
    print("Final output of the workflow:")
    print(completed_steps[-1] if completed_steps else "No steps were completed.")


if __name__ == "__main__":
    main()

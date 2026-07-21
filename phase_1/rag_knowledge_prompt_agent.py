"""Exercises the provided RAGKnowledgePromptAgent end to end:
chunk → embed → retrieve → answer from the best chunk only."""
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import RAGKnowledgePromptAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

rag_agent = RAGKnowledgePromptAgent(openai_api_key, "a helpful research assistant")

corpus = (
    "Clara Barton founded the American Red Cross in 1881 after serving as a "
    "nurse during the American Civil War, where she earned the nickname "
    "'Angel of the Battlefield'. She led the organization for 23 years. "
    "Separately, the Eiffel Tower was completed in 1889 for the World's Fair "
    "in Paris and stands 330 meters tall. It was designed by Gustave Eiffel's "
    "engineering company."
)

rag_agent.chunk_text(corpus)
rag_agent.calculate_embeddings()

prompt = "Who founded the American Red Cross and when?"
print(f"Prompt: {prompt}\n")
print(rag_agent.find_prompt_in_knowledge(prompt))

"""Agent library for building agentic workflows (Phase 1).

Seven agent patterns, from a bare LLM passthrough up to routing and
action planning. Each class keeps a single responsibility so they can
be composed freely in Phase 2.
"""

import csv
import json
import re
import uuid
from typing import Protocol

import numpy as np
import pandas as pd
from openai import OpenAI

CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-large"


class WorkerAgent(Protocol):
    """What an agent needs to look like to be evaluated or routed to."""

    def respond(self, prompt: str) -> str: ...


def _default_client(api_key):
    return OpenAI(api_key=api_key, max_retries=8)


def _cosine(u, v):
    """Cosine similarity between two vectors (lists or arrays)."""
    a, b = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class DirectPromptAgent:
    """Bare passthrough: user prompt in, model text out.

    No system prompt, no memory, no tools — the baseline against which
    every other agent pattern can be compared.
    """

    def __init__(self, openai_api_key, client=None, chat_model=CHAT_MODEL):
        self.openai_api_key = openai_api_key
        self.chat_model = chat_model
        self._client = client or _default_client(openai_api_key)

    def respond(self, prompt):
        completion = self._client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        # Hand back just the text, never the raw payload
        return completion.choices[0].message.content


class AugmentedPromptAgent:
    """Adds a persona on top of the direct pattern via a system prompt."""

    def __init__(self, openai_api_key, persona, client=None, chat_model=CHAT_MODEL):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.chat_model = chat_model
        self._client = client or _default_client(openai_api_key)

    def respond(self, input_text):
        system_msg = f"You are {self.persona}. Forget all previous context."
        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    """Persona + a closed knowledge base.

    The system prompt fences the model into the supplied knowledge so
    answers come from what we injected, not from pretraining.
    """

    def __init__(self, openai_api_key, persona, knowledge, client=None, chat_model=CHAT_MODEL):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.chat_model = chat_model
        self._client = client or _default_client(openai_api_key)

    def respond(self, input_text):
        # Wording below is intentionally kept as specified.
        fence = (
            f"You are {self.persona} knowledge-based assistant. "
            "Forget all previous context. "
            "Use only the following knowledge to answer, do not use your own knowledge: "
            f"{self.knowledge} "
            "Answer the prompt based on this knowledge, not your own."
        )
        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": fence},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class RAGKnowledgePromptAgent:
    """Retrieval-augmented agent: chunk a corpus, embed it, and answer
    prompts from the single most similar chunk.

    (Provided as part of the starter material; kept here so the package
    imports as a whole. Prefer the starter repo's copy if it differs.)
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100,
                 client=None, chat_model=CHAT_MODEL, embedding_model=EMBEDDING_MODEL):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.unique_filename = f"{uuid.uuid4()}.csv"
        self._client = client or _default_client(openai_api_key)

    def get_embedding(self, text):
        result = self._client.embeddings.create(
            model=self.embedding_model, input=text, encoding_format="float"
        )
        return result.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        return _cosine(vector_one, vector_two)

    def chunk_text(self, text):
        text = re.sub(r"\s+", " ", text)
        newline = "\n"

        if len(text) <= self.chunk_size:
            pieces = [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
        else:
            pieces, cursor, idx = [], 0, 0
            while cursor < len(text):
                stop = min(cursor + self.chunk_size, len(text))
                window = text[cursor:stop]
                if newline in window:
                    stop = cursor + window.rindex(newline) + len(newline)
                pieces.append(
                    {
                        "chunk_id": idx,
                        "text": text[cursor:stop],
                        "chunk_size": stop - cursor,
                        "start_char": cursor,
                        "end_char": stop,
                    }
                )
                cursor = stop - self.chunk_overlap
                idx += 1

        with open(f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for piece in pieces:
                writer.writerow({"text": piece["text"], "chunk_size": piece["chunk_size"]})
        return pieces

    def calculate_embeddings(self):
        table = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        table["embeddings"] = table["text"].apply(self.get_embedding)
        table.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return table

    def find_prompt_in_knowledge(self, prompt):
        query_vec = self.get_embedding(prompt)
        table = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        table["embeddings"] = table["embeddings"].apply(lambda s: np.array(json.loads(s)))
        table["similarity"] = table["embeddings"].apply(lambda v: _cosine(query_vec, v))
        top_chunk = table.loc[table["similarity"].idxmax(), "text"]

        answer = self._client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}, a knowledge-based assistant. "
                        "Forget previous context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information: {top_chunk}. Prompt: {prompt}",
                },
            ],
            temperature=0,
        )
        return answer.choices[0].message.content


class EvaluationAgent:
    """Quality gate around a worker agent.

    Runs a bounded critique loop: ask the worker, judge the answer
    against fixed criteria, and — when the judge says no — turn the
    verdict into concrete correction instructions for the next round.
    """

    def __init__(self, openai_api_key, persona, evaluation_criteria,
                 agent_to_evaluate: WorkerAgent, max_interactions,
                 client=None, chat_model=CHAT_MODEL):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions
        self.chat_model = chat_model
        self._client = client or _default_client(openai_api_key)

    def _judge(self, user_content):
        """One temperature-0 call in the evaluator's own persona."""
        reply = self._client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}. Forget all previous context.",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        return reply.choices[0].message.content

    def evaluate(self, initial_prompt):
        working_prompt = initial_prompt
        worker_answer = ""
        verdict = ""
        rounds_used = 0

        for round_no in range(1, self.max_interactions + 1):
            rounds_used = round_no
            print(f"\n--- Interaction {round_no} ---")

            # 1) Worker produces an answer
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{working_prompt}")
            worker_answer = self.agent_to_evaluate.respond(working_prompt)
            print(f"Worker Agent Response:\n{worker_answer}")

            # 2) Judge it against the criteria
            print(" Step 2: Evaluator agent judges the response")
            verdict = self._judge(
                f"Does the following answer meet this criteria: {self.evaluation_criteria}\n"
                f"Answer to evaluate:\n{worker_answer}\n"
                "Respond with 'Yes' or 'No', followed by a brief explanation."
            )
            print(f"Evaluator Agent Evaluation:\n{verdict}")

            # 3) Accept, or convert the verdict into fix instructions
            if verdict.strip().lower().startswith("yes"):
                print("Final solution accepted.")
                break

            print(" Step 3: Evaluator agent generates correction instructions")
            fixes = self._judge(
                "Provide concise instructions to fix an answer based on these "
                f"reasons why it is incorrect:\n{verdict}"
            )
            print(f"Instructions to fix:\n{fixes}")

            working_prompt = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {worker_answer}\n"
                "It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {fixes}"
            )

        return {
            "final_response": worker_answer,
            "evaluation": verdict,
            "iterations": rounds_used,
        }


class RoutingAgent:
    """Semantic dispatcher.

    Holds a registry of agents (name, description, callable) and sends
    each incoming prompt to whichever description sits closest to it in
    embedding space.
    """

    def __init__(self, openai_api_key, agents=None, client=None,
                 embedding_model=EMBEDDING_MODEL):
        self.openai_api_key = openai_api_key
        self.agents = agents if agents is not None else []
        self.embedding_model = embedding_model
        self._client = client or _default_client(openai_api_key)

    def get_embedding(self, text):
        result = self._client.embeddings.create(
            model=self.embedding_model, input=text, encoding_format="float"
        )
        return result.data[0].embedding

    def route(self, user_input):
        prompt_vec = self.get_embedding(user_input)

        winner, top_score = None, float("-inf")
        for candidate in self.agents:
            score = _cosine(prompt_vec, self.get_embedding(candidate["description"]))
            print(f"Similarity with '{candidate['name']}': {score:.4f}")
            if score > top_score:
                winner, top_score = candidate, score

        if winner is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {winner['name']} (score={top_score:.4f})")
        return winner["func"](user_input)


class ActionPlanningAgent:
    """Turns a task described in natural language into an ordered list
    of executable steps, drawing on the knowledge it was given."""

    # Matches leading list markers: "1.", "2)", "-", "*", "Step 3:" etc.
    _MARKER = re.compile(r"^(?:step\s*\d+\s*[:.)-]?|\d+\s*[.)-]|[-*•])\s*", re.IGNORECASE)

    def __init__(self, openai_api_key, knowledge, client=None, chat_model=CHAT_MODEL):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.chat_model = chat_model
        self.client = client or _default_client(openai_api_key)

    def extract_steps_from_prompt(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Action Planning Agent. Using your knowledge, "
                        "you extract from the user prompt the steps requested to "
                        "complete the action the user is asking for. You return the "
                        "steps as a list. Only return the steps in your knowledge. "
                        "Forget any previous context. This is your knowledge: "
                        f"{self.knowledge}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = completion.choices[0].message.content

        steps = []
        for raw_line in raw.splitlines():
            stripped = raw_line.strip()
            match = self._MARKER.match(stripped)
            if not match:
                continue  # not an actual list item — a preamble/header line
            line = stripped[match.end():].strip()
            if line:
                steps.append(line)
        return steps

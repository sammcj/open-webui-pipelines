"""
title: MoA Pipeline
author: Sam McLeod
author_url: https://github.com/sammcj
funding_url: https://github.com/sammcj
version: 0.1
"""

# requirements: pydantic asyncio typing

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import os
import requests
import json
import asyncio
from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
)


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = Field(
            default=["*"],
            description="List of target pipeline ids this filter will connect to",
        )
        priority: int = Field(
            default=0, description="Priority level for filter execution"
        )
        OPENAI_API_BASE_URL: str = Field(
            default="http://host.docker.internal:11434/v1",
            description="OpenAI Compatible API base URL",
        )
        OPENAI_API_KEY: str = Field(..., description="OpenAI Compatible API key")
        TASK_MODEL: str = Field(
            default="llama3.1:8b-instruct-q6_K",
            description="Model for task decomposition",
        )
        AGENT_MODELS: Dict[str, str] = Field(
            default={
                "default": "llama3.1:8b-instruct-q6_K",
                "research": "llama3.1:34b-instruct-q6_K",
                "creativity": "llama3.1:34b-code-instruct-q6_K",
            },
            description="Models for individual agents with their roles",
        )
        AGENT_ROLES: List[str] = Field(
            default=["default", "research", "creativity"],
            description="Roles assigned to agents in order",
        )
        AGGREGATOR_MODEL: str = Field(
            default="llama3.1:34b-instruct-q6_K",
            description="Model for result aggregation (usually a larger model)",
        )
        NUM_AGENTS: int = Field(default=3, description="Number of agents to use")
        MAX_ITERATIONS: int = Field(
            default=3, description="Maximum number of iteration rounds"
        )
        SIMILARITY_THRESHOLD: float = Field(
            default=0.8,
            description="Threshold for response similarity to stop iterations",
        )
        CONTEXT_TEMPLATE: str = Field(
            default="""Use the following context as your learned knowledge:
<context>
{{CONTEXT}}
</context>

When answering:
- Always use British English spelling.
- If you're unsure, ask for clarification.
Avoid mentioning that you obtained information from the context.""",
            description="Template for context injection",
        )

    def __init__(self):
        self.type = "filter"
        self.name = "Enhanced Mixture of Agents (MoA) Pipeline"
        self.valves = self.Valves(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "ollama"),
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def call_openai_api(
        self, messages: List[Dict[str, str]], model: str, **kwargs
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
        }

        for param in ["temperature", "top_p", "top_k", "max_tokens"]:
            if param in kwargs:
                payload[param] = kwargs[param]

        response = requests.post(
            f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def task_decomposition(self, user_message: str, **kwargs) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "Break down the user's task into smaller, focused subtasks.",
            },
            {
                "role": "user",
                "content": f"Decompose this task into {self.valves.NUM_AGENTS} subtasks: {user_message}",
            },
        ]
        response = await self.call_openai_api(
            messages, self.valves.TASK_MODEL, **kwargs
        )
        return [task.strip() for task in response.split("\n") if task.strip()]

    async def agent_execution(
        self, subtask: str, context: str, agent_id: int, **kwargs
    ) -> str:
        role = self.valves.AGENT_ROLES[agent_id % len(self.valves.AGENT_ROLES)]
        model = self.valves.AGENT_MODELS.get(role, self.valves.AGENT_MODELS["default"])
        messages = [
            {
                "role": "system",
                "content": f"You are Agent {agent_id} with the role of {role}. {self.valves.CONTEXT_TEMPLATE.replace('{{CONTEXT}}', context)}",
            },
            {"role": "user", "content": subtask},
        ]
        return await self.call_openai_api(messages, model, **kwargs)

    async def result_aggregation(
        self, user_message: str, agent_responses: List[str], iteration: int, **kwargs
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": f"You are the Aggregator. Synthesize the agents' responses into a coherent answer. This is iteration {iteration}.",
            },
            {
                "role": "user",
                "content": f"Original question: {user_message}\n\nAgent responses:\n"
                + "\n".join(agent_responses),
            },
        ]
        return await self.call_openai_api(
            messages, self.valves.AGGREGATOR_MODEL, **kwargs
        )

    def calculate_similarity(self, response1: str, response2: str) -> float:
        return len(set(response1.split()) & set(response2.split())) / len(
            set(response1.split() + response2.split())
        )

    async def moa_process(self, user_message: str, **kwargs) -> str:
        subtasks = await self.task_decomposition(user_message, **kwargs)
        previous_response = ""

        for iteration in range(self.valves.MAX_ITERATIONS):
            agent_responses = await asyncio.gather(
                *[
                    self.agent_execution(subtask, "", i, **kwargs)
                    for i, subtask in enumerate(subtasks)
                ]
            )
            aggregated_response = await self.result_aggregation(
                user_message, agent_responses, iteration + 1, **kwargs
            )

            if iteration > 0:
                similarity = self.calculate_similarity(
                    previous_response, aggregated_response
                )
                if similarity >= self.valves.SIMILARITY_THRESHOLD:
                    break

            previous_response = aggregated_response

        return aggregated_response

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if body.get("title", False):
            return body

        print(f"pipe:{__name__}")
        user_message = get_last_user_message(body["messages"])

        # Extract additional parameters from the body only if they are present
        kwargs = {}
        for param in ["temperature", "top_p", "top_k", "max_tokens"]:
            if param in body:
                kwargs[param] = body[param]

        moa_response = await self.moa_process(user_message, **kwargs)

        system_prompt = self.valves.CONTEXT_TEMPLATE.replace(
            "{{CONTEXT}}", moa_response
        )
        messages = add_or_update_system_message(system_prompt, body["messages"])

        return {**body, "messages": messages}

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

"""
Title: Mixture of Agents Action
Author: Sam McLeod
Version: 0.1
required_open_webui_version: 0.3.9
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Awaitable
import aiohttp
import asyncio
import random
import time
import uuid


class Action:
    class Valves(BaseModel):
        models: Optional[List[str]] = Field(
            default=["llama3.1:8b-instruct-q6_K", "codegeex4:9b-all-q6_K"],
            description="List of models to use in the MoA architecture.",
        )
        aggregator_model: str = Field(
            default="llama3.1:8b-instruct-q6_K",
            description="Model to use for aggregation tasks.",
        )
        ollama_api_base: str = Field(
            default="http://ollama:11434/v1",
            description="Base URL for Ollama API.",
        )
        num_layers: int = Field(default=2, description="Number of MoA layers.")
        num_agents_per_layer: int = Field(
            default=1, description="Number of agents to use in each layer."
        )
        emit_interval: float = Field(
            default=1.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )
        timeout: float = Field(
            default=60.0, description="Timeout for API requests in seconds"
        )
        max_retries: int = Field(
            default=3, description="Maximum number of retries for failed API requests"
        )
        show_individual_responses: bool = Field(
            default=False, description="Show individual model responses in the output"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.type = "filter"
        self.name = "Mixture of Agents Action"

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Optional[dict]:
        try:
            print("Starting MoA action")
            await self.emit_status(
                __event_emitter__, "info", "Starting Mixture of Agents process", False
            )

            messages = body.get("messages", [])
            if not messages:
                raise ValueError("No messages found in the request body")

            last_message = messages[-1]["content"]
            print(f"Last message: {last_message}")

            moa_response = await self.moa_process(last_message, __event_emitter__)
            print(f"MoA response: {moa_response}")

            # Create a new message with the MoA response
            new_message = {
                "role": "assistant",
                "content": moa_response,
                "id": str(uuid.uuid4()),
                "name": "MoA_Assistant",
            }

            # Append the new message to the conversation
            messages.append(new_message)

            # Update the body with the modified messages
            body["messages"] = messages

            print(f"Updated messages: {body['messages']}")

            # If there's a history object, update it as well
            if "history" in body:
                if "messages" not in body["history"]:
                    body["history"]["messages"] = {}
                body["history"]["messages"][new_message["id"]] = new_message
                body["history"]["currentId"] = new_message["id"]

            # Update the UI with the new message
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "update_response",
                        "data": {
                            "messages": messages,
                            "history": body.get("history"),
                        },
                    }
                )

            print("MoA action completed")
            return body

        except Exception as e:
            error_msg = f"Error in Mixture of Agents process: {str(e)}"
            print(f"Error: {error_msg}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "error",
                        "data": {"message": error_msg},
                    }
                )
            return {"error": error_msg}

    async def outlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        print(f"outlet:{__name__}")
        print(f"Outlet body: {body}")

        # Ensure the messages are updated in the body
        messages = body.get("messages", [])
        if messages:
            last_message = messages[-1]
            print(f"Last message in outlet: {last_message}")

            # If there's an event emitter, use it to update the UI
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "update_response",
                        "data": {
                            "messages": messages,
                            "history": body.get("history"),
                        },
                    }
                )
                print("Emitted update_response event")
            else:
                print("No __event_emitter__ available in outlet")
        else:
            print("No messages found in outlet body")

        return body

    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "in_progress",
                            "message": "Starting Mixture of Agents process",
                        },
                    }
                )

            messages = body.get("messages", [])
            if not messages:
                raise ValueError("No messages found in the request body")

            last_message = messages[-1]["content"]
            moa_response = await self.moa_process(last_message, __event_emitter__)

            # Format the MoA response
            moa_formatted_response = f"[Mixture of Agents Response]\n{moa_response}"

            # Add the MoA response as a new message
            moa_message = {
                "role": "assistant",
                "content": moa_formatted_response,
                "name": "MoA_Assistant",
            }

            # Add the new MoA message to the conversation
            updated_messages = messages + [moa_message]

            # Update the body with the new messages
            updated_body = body.copy()
            updated_body["messages"] = updated_messages

            # Emit the updated response
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "update_response", "data": {"messages": updated_messages}}
                )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "complete",
                            "message": "Mixture of Agents process completed",
                        },
                    }
                )

            return updated_body

        except Exception as e:
            error_msg = f"Error in Mixture of Agents process: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"status": "error", "message": error_msg},
                    }
                )
            return {"error": error_msg}

    async def validate_models(
        self, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ):
        await self.emit_status(__event_emitter__, "info", "Validating models", False)
        valid_models = []
        for model in self.valves.models:
            response = await self.query_ollama(model, "Test prompt", __event_emitter__)
            if isinstance(response, str) and not response.startswith("Error:"):
                valid_models.append(model)

        if not valid_models:
            raise ValueError(
                "No valid models available. Please check your model configurations."
            )

        self.valves.models = valid_models
        await self.emit_status(
            __event_emitter__, "info", f"Validated {len(valid_models)} models", False
        )

    async def moa_process(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        try:
            print(f"Starting MOA process with prompt: {prompt}")
            if len(self.valves.models) < self.valves.num_agents_per_layer:
                raise ValueError(
                    f"Not enough models available. Required: {self.valves.num_agents_per_layer}, Available: {len(self.valves.models)}"
                )

            layer_outputs = []
            for layer in range(self.valves.num_layers):
                print(f"Processing layer {layer + 1}/{self.valves.num_layers}")
                await self.emit_status(
                    __event_emitter__,
                    "info",
                    f"Processing layer {layer + 1}/{self.valves.num_layers}",
                    False,
                )

                layer_agents = random.sample(
                    self.valves.models,
                    self.valves.num_agents_per_layer,
                )
                print(f"Selected agents for layer {layer + 1}: {layer_agents}")

                tasks = [
                    self.process_agent(
                        prompt, agent, layer, i, layer_outputs, __event_emitter__
                    )
                    for i, agent in enumerate(layer_agents)
                ]
                current_layer_outputs = await asyncio.gather(*tasks)
                print(
                    f"Received outputs for layer {layer + 1}: {current_layer_outputs}"
                )

                valid_outputs = [
                    output
                    for output in current_layer_outputs
                    if isinstance(output, str) and not output.startswith("Error:")
                ]
                if not valid_outputs:
                    raise ValueError(
                        f"No valid responses received from any agent in layer {layer + 1}"
                    )

                layer_outputs.append(valid_outputs)
                print(f"Added valid outputs to layer_outputs: {valid_outputs}")
                await self.emit_status(
                    __event_emitter__,
                    "info",
                    f"Completed layer {layer + 1}/{self.valves.num_layers}",
                    False,
                )

            print("Creating final aggregator prompt")
            await self.emit_status(
                __event_emitter__, "info", "Creating final aggregator prompt", False
            )
            final_prompt = self.create_final_aggregator_prompt(prompt, layer_outputs)
            print(f"Final aggregator prompt: {final_prompt}")

            print("Generating final response")
            await self.emit_status(
                __event_emitter__, "info", "Generating final response", False
            )
            final_response = await self.query_ollama(
                self.valves.aggregator_model, final_prompt, __event_emitter__
            )
            print(f"Received final response: {final_response}")

            if isinstance(final_response, str) and final_response.startswith("Error:"):
                raise ValueError(
                    f"Failed to generate final response. Last error: {final_response}"
                )

            print("MOA process completed successfully")
            return final_response

        except Exception as e:
            error_msg = f"Error in MOA process: {str(e)}"
            print(error_msg)
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            return f"Error in MOA process: {str(e)}"

    async def process_agent(
        self, prompt, agent, layer, agent_index, layer_outputs, __event_emitter__
    ):
        await self.emit_status(
            __event_emitter__,
            "info",
            f"Querying agent {agent_index + 1} in layer {layer + 1}",
            False,
        )

        if layer == 0:
            response = await self.query_ollama(agent, prompt, __event_emitter__)
        else:
            await self.emit_status(
                __event_emitter__,
                "info",
                f"Creating aggregator prompt for layer {layer + 1}",
                False,
            )
            aggregator_prompt = self.create_aggregator_prompt(prompt, layer_outputs[-1])
            response = await self.query_ollama(
                self.valves.aggregator_model, aggregator_prompt, __event_emitter__
            )

        await self.emit_status(
            __event_emitter__,
            "info",
            f"Received response from agent {agent_index + 1} in layer {layer + 1}",
            False,
        )
        return response

    def create_aggregator_prompt(
        self, original_prompt: str, previous_responses: List[str]
    ) -> str:
        aggregator_prompt = (
            f"Original prompt: {original_prompt}\n\nPrevious responses:\n"
        )
        for i, response in enumerate(previous_responses, 1):
            aggregator_prompt += f"{i}. {response}\n\n"
        aggregator_prompt += "Based on the above responses and the original prompt, provide an improved and comprehensive answer:"
        return aggregator_prompt

    def create_final_aggregator_prompt(
        self, original_prompt: str, all_layer_outputs: List[List[str]]
    ) -> str:
        final_prompt = (
            f"Original prompt: {original_prompt}\n\nResponses from all layers:\n"
        )
        for layer, responses in enumerate(all_layer_outputs, 1):
            final_prompt += f"Layer {layer}:\n"
            for i, response in enumerate(responses, 1):
                final_prompt += f" {i}. {response}\n\n"
        final_prompt += (
            "Considering all the responses from different layers and the original prompt, provide a final, comprehensive answer that strictly adheres to the original request:\n"
            "1. Incorporate relevant information from all previous responses seamlessly.\n"
            "2. Avoid referencing or acknowledging previous responses explicitly unless directed by the prompt.\n"
            "3. Provide a complete and detailed reply addressing the original prompt."
        )
        return final_prompt

    async def query_ollama(
        self,
        model: str,
        prompt: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        url = f"{self.valves.ollama_api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}]}

        for attempt in range(self.valves.max_retries):
            try:
                await self.emit_status(
                    __event_emitter__,
                    "info",
                    f"Sending API request to model: {model} (Attempt {attempt + 1})",
                    False,
                )

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, json=data, timeout=self.valves.timeout
                    ) as response:
                        if response.status == 404:
                            raise ValueError(
                                f"Model '{model}' not found. Please check if the model is available and correctly specified."
                            )

                        response.raise_for_status()
                        result = await response.json()

                await self.emit_status(
                    __event_emitter__,
                    "info",
                    f"Received API response from model: {model}",
                    False,
                )

                return result["choices"][0]["message"]["content"]
            except aiohttp.ClientResponseError as e:
                error_message = f"HTTP error querying Ollama API for model {model}: {e.status}, {e.message}"
                if attempt == self.valves.max_retries - 1:
                    await self.emit_status(
                        __event_emitter__, "error", error_message, True
                    )
                    return f"Error: Unable to query model {model} due to HTTP error {e.status}"
            except aiohttp.ClientError as e:
                error_message = (
                    f"Network error querying Ollama API for model {model}: {str(e)}"
                )
                if attempt == self.valves.max_retries - 1:
                    await self.emit_status(
                        __event_emitter__, "error", error_message, True
                    )
                    return f"Error: Unable to query model {model} due to network error"
            except asyncio.TimeoutError:
                error_message = f"Timeout error querying Ollama API for model {model}"
                if attempt == self.valves.max_retries - 1:
                    await self.emit_status(
                        __event_emitter__, "error", error_message, True
                    )
                    return f"Error: Unable to query model {model} due to timeout"
            except Exception as e:
                error_message = (
                    f"Unexpected error querying Ollama API for model {model}: {str(e)}"
                )
                if attempt == self.valves.max_retries - 1:
                    await self.emit_status(
                        __event_emitter__, "error", error_message, True
                    )
                    return (
                        f"Error: Unable to query model {model} due to unexpected error"
                    )

            await asyncio.sleep(2**attempt)  # Exponential backoff

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    async def on_start(self):
        print("Mixture of Agents Action started")

    async def on_stop(self):
        print("Mixture of Agents Action stopped")

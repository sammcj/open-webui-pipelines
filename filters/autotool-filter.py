"""
title: AutoTool
author: Sam McLeod, Wes Caldwell, Joshua Jama @ Perplexity
date: 2025-03-07
version: 4.1
license: MIT
description: Automatically recommends tools based on the users prompt.
git: https://github.com/sammcj/open-webui-pipelines/blob/main/filters/autotool-filter.py
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, List, Dict, Set
import json
import re
import logging
from fastapi import Request

# Import paths for Open WebUI 0.5
from open_webui.models.users import Users
from open_webui.models.tools import Tools
from open_webui.models.models import Models

# Updated import for Open WebUI 0.5
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("autotool-filter")


class Filter:
    class Valves(BaseModel):
        template: str = Field(
            default="""Tools: {{TOOLS}}

Available tools:
{{TOOL_DETAILS}}

You are an expert at selecting the most appropriate tools based on user queries.

If no tool matches the query, return an empty list [].
Otherwise, return a list of matching tool IDs in the format ["tool_id"].
Select multiple tools if they would work together to solve the user's request.
Only return the list. Do not return any other text.
Review the entire chat history to ensure the selected tool matches the context.
If unsure, default to an empty list [].
Use tools conservatively - only select tools that directly address the user's needs."""
        )
        status: bool = Field(default=False)
        use_semantic_matching: bool = Field(
            default=False,
            description="Enable semantic similarity matching for tool selection",
        )
        semantic_threshold: float = Field(
            default=0.5,
            description="Threshold for semantic similarity matching (0.0 to 1.0)",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_history = {}
        self.tool_analytics = {}
        self.user_feedback = {}
        self.models = {}
        self.semantic_model = None
        pass

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Request = None,
    ) -> dict:
        messages = body["messages"]
        user_message = get_last_user_message(messages)

        logger.info("Processing AutoTool filter request")

        if self.valves.status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Finding the right tools...",
                        "done": False,
                    },
                }
            )

        # Initialise semantic model if enabled and not already loaded
        if self.valves.use_semantic_matching and self.semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading semantic model for tool matching")
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Semantic model loaded successfully")
            except ImportError:
                logger.warning(
                    "Semantic matching enabled but sentence-transformers not available"
                )
                self.valves.use_semantic_matching = False

        all_tools = [
            {"id": tool.id, "description": tool.meta.description}
            for tool in Tools.get_tools()
        ]
        available_tool_ids = (
            __model__.get("info", {}).get("meta", {}).get("toolIds", [])
        )
        available_tools = [
            tool for tool in all_tools if tool["id"] in available_tool_ids
        ]

        logger.debug(f"Available tools: {len(available_tools)}")

        # Tool Recommendation Based on User History
        user_id = __user__["id"]
        if user_id in self.user_history:
            recommended_tools = [
                tool
                for tool in available_tools
                if tool["id"] in self.user_history[user_id]
            ]
            # Add other available tools to ensure we don't miss any
            for tool in available_tools:
                if tool not in recommended_tools:
                    recommended_tools.append(tool)
        else:
            recommended_tools = available_tools

        # Dynamic Tool Filtering
        if self.valves.use_semantic_matching and self.semantic_model is not None:
            filtered_tools = self.semantic_filter_tools(recommended_tools, user_message)
            logger.info(
                f"Semantic filtering found {len(filtered_tools)} potential tools"
            )
        else:
            filtered_tools = self.filter_tools(recommended_tools, user_message)
            logger.info(f"Basic filtering found {len(filtered_tools)} potential tools")

        # Create detailed tool descriptions for the prompt
        tool_details = "\n".join(
            [f"- {tool['id']}: {tool['description']}" for tool in filtered_tools]
        )

        system_prompt = self.valves.template.replace("{{TOOLS}}", str(filtered_tools))
        system_prompt = system_prompt.replace("{{TOOL_DETAILS}}", tool_details)

        prompt = (
            "History:\n"
            + "\n".join(
                [
                    f"{message['role'].upper()}: \"\"\"{message['content']}\"\"\""
                    for message in messages[::-1][:4]
                ]
            )
            + f"\nQuery: {user_message}"
        )

        payload = {
            "model": body["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        try:
            user = Users.get_user_by_id(user_id)
            logger.info(f"Generating chat completion for user {user_id}")
            response = await generate_chat_completion(
                __request__, form_data=payload, user=user
            )
            content = response["choices"][0]["message"]["content"]

            # Parse the function response
            if content is not None:
                logger.debug(f"LLM response content: {content}")
                result = self.parse_tool_response(content)

                if isinstance(result, list) and len(result) > 0:

                    body["tool_ids"] = result
                    if self.valves.status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Found matching tools: {', '.join(result)}",
                                    "done": True,
                                },
                            }
                        )
                    # Update Tool Usage Analytics
                    self.update_tool_analytics(result)
                    # Update User History
                    self.update_user_history(user_id, result)
                    logger.info(f"Selected tools: {result}")
                else:
                    if self.valves.status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "No matching tools found.",
                                    "done": True,
                                },
                            }
                        )
                    logger.info("No matching tools found")

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error processing request: {str(e)}",
                            "done": True,
                        },
                    }
                )
            pass

        return body

    def filter_tools(self, tools: List[Dict], query: str) -> List[Dict]:
        """Enhanced tool filtering with better matching algorithms"""
        filtered_tools = []
        query_lower = query.lower()

        # Extract key terms from the query
        # This is a simple approach - could be enhanced with NLP libraries
        query_terms = set(query_lower.split())

        for tool in tools:
            description_lower = tool["description"].lower()
            tool_id_lower = tool["id"].lower()

            # Direct match in description (original implementation)
            if query_lower in description_lower:
                filtered_tools.append(tool)
                continue

            # Check for tool ID match or partial match
            if query_lower in tool_id_lower or any(
                term in tool_id_lower for term in query_terms
            ):
                filtered_tools.append(tool)
                continue

            # Check for keyword matches in description
            if any(term in description_lower for term in query_terms):
                filtered_tools.append(tool)
                continue

        # If no tools matched, return all tools to let the LLM decide
        if not filtered_tools:
            return tools

        return filtered_tools

    def semantic_filter_tools(self, tools: List[Dict], query: str) -> List[Dict]:
        """Filter tools based on semantic similarity"""
        if not self.semantic_model or not self.valves.use_semantic_matching:
            return self.filter_tools(tools, query)

        try:
            query_embedding = self.semantic_model.encode(query)

            results = []
            for tool in tools:
                description = tool["description"]
                desc_embedding = self.semantic_model.encode(description)

                # Calculate cosine similarity
                from scipy import spatial

                similarity = 1 - spatial.distance.cosine(
                    query_embedding, desc_embedding
                )

                logger.debug(f"Tool {tool['id']} similarity: {similarity}")

                if similarity > self.valves.semantic_threshold:
                    results.append(tool)

            # If no tools matched the threshold, return all tools to let the LLM decide
            if not results:
                return tools

            return results
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}", exc_info=True)
            # Fall back to basic filtering if there's an error
            return self.filter_tools(tools, query)

    def parse_tool_response(self, content: str) -> List[str]:
        """Parse the LLM response to extract tool IDs with enhanced error handling"""
        if not content:
            return []

        # Try multiple parsing approaches
        result = []

        # Approach 1: Direct JSON parsing
        try:
            content_normalised = content.replace("'", '"')
            result = json.loads(content_normalised)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed, trying alternative approaches")

        # Approach 2: Extract array with regex
        try:
            json_array_match = re.search(r"\[(.*?)\]", content)
            if json_array_match:
                json_str = f"[{json_array_match.group(1)}]"
                json_str = json_str.replace("'", '"')
                result = json.loads(json_str)
                if isinstance(result, list):
                    return result
        except (json.JSONDecodeError, AttributeError):
            logger.debug("Regex JSON extraction failed, trying next approach")

        # Approach 3: Extract quoted strings as individual tools
        try:
            quoted_strings = re.findall(r'["\'](.*?)["\']', content)
            if quoted_strings:
                return quoted_strings
        except Exception as e:
            logger.debug(f"Quoted string extraction failed: {e}")

        return []

    def update_user_history(self, user_id: str, tool_ids: List[str]) -> None:
        """Update user history with selected tools"""
        if user_id in self.user_history:
            # Update existing history
            for tool_id in tool_ids:
                if tool_id not in self.user_history[user_id]:
                    self.user_history[user_id].append(tool_id)
        else:
            # Create new history entry
            self.user_history[user_id] = tool_ids

    def update_tool_analytics(self, tool_ids: List[str]) -> None:
        """Update analytics for tool usage"""
        for tool_id in tool_ids:
            if tool_id in self.tool_analytics:
                self.tool_analytics[tool_id] += 1
            else:
                self.tool_analytics[tool_id] = 1
        logger.debug(f"Updated tool analytics: {self.tool_analytics}")

    def integrate_user_feedback(self, user_id: str, feedback: Dict) -> None:
        """Store user feedback for tool recommendations"""
        if user_id in self.user_feedback:
            self.user_feedback[user_id].append(feedback)
        else:
            self.user_feedback[user_id] = [feedback]
        logger.info(f"Integrated user feedback for user {user_id}")

    def add_model(self, model_id: str, model_info: Dict) -> None:
        """Add model information to the filter's model registry"""
        self.models[model_id] = model_info
        logger.debug(f"Added model {model_id} to registry")

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Retrieve model information from the filter's model registry"""
        return self.models.get(model_id)

"""
title: DeepSeek r1 Formatter
author: sammcj
author_url:
version: 0.1
"""

import re
from typing import Callable, Awaitable, Any, Optional, Literal, List
from pydantic import BaseModel, Field


def extract_thinking_content(message: str):
    match = re.search(r"<think>(.*?)</think>", message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_output_content(message: str):
    match = re.search(r"</think>(.*)", message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def replace_message(self, message):
        await self.event_emitter({"type": "replace", "data": {"content": message}})

    async def outlet(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
    ) -> dict:
        self.event_emitter = __event_emitter__

        if len(body["messages"]) == 0:
            return body

        last_reply: dict = body["messages"][-1]
        last_reply_content = last_reply["content"].strip()

        thinking_content = extract_thinking_content(last_reply_content)
        output_content = extract_output_content(last_reply_content)

        new_message = ""

        # Replace newlines in thinking_content and add '> ' at the start of each line
        thinking_content = thinking_content.replace("\n", "\n> ")
        thinking_content = "> " + thinking_content
        new_message += f"<details>\n<summary>Deep Thinking</summary>\n{thinking_content}\n</details>\n"

        if output_content:
            new_message += f"{output_content}"

        if not output_content and not thinking_content:
            new_message = (
                last_reply_content  # Use original content if no tags are present
            )

        if new_message:
            # Remove any extra newline at the end
            new_message = new_message.rstrip("\n")

        if new_message != last_reply_content:
            body["messages"][-1]["content"] = new_message
            await self.replace_message(new_message)

        return body

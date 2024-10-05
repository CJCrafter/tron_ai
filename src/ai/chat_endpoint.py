from abc import ABC, abstractmethod
from typing import Dict, List, Literal

import openai

from src.settings import SETTINGS


class ChatEndpoint(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        pass


class OpenAIChatEndpoint(ChatEndpoint):
    def __init__(
        self,
        client: openai.AsyncClient = openai.AsyncClient(api_key=SETTINGS.openai_api_key),
        model: str = "gpt-4o-mini",
    ):
        self._async_client = client
        self.model = model

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        response = await self._async_client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=temperature,
            response_format={"type": response_format},
        )
        return response.choices[0].message.content

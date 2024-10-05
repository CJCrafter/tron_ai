from abc import ABC, abstractmethod
from typing import List

import openai

from src.settings import SETTINGS


class EmbeddingsEndpoint(ABC):
    @abstractmethod
    async def embed(self, text: List[str]) -> List[float]:
        pass


class OpenAIEmbeddingsEndpoint(EmbeddingsEndpoint):
    def __init__(
        self,
        client: openai.AsyncClient = openai.AsyncClient(api_key=SETTINGS.openai_api_key),
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ):
        self._async_client = client
        self.model = model
        self.dimensions = dimensions

    async def embed(self, text: List[str]) -> List[List[float]]:
        response = await self._async_client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimensions,
        )
        return [element.embedding for element in response.data]

from pathlib import Path

import yaml

from src.ai.chat_endpoint import ChatEndpoint, OpenAIChatEndpoint
from src.ai.embeddings_endpoint import OpenAIEmbeddingsEndpoint, EmbeddingsEndpoint
from src.ai.prompt import Prompt
from src.util.register import TypedRegisterRegister, Register

REGISTERS = TypedRegisterRegister()


#############################
# Chat Generation Endpoints #
#############################
_chat_endpoints = Register[ChatEndpoint]({
    "gpt-4o-mini": OpenAIChatEndpoint(model="gpt-4o-mini"),
    "o1-preview": OpenAIChatEndpoint(model="o1-preview"),
})
REGISTERS[ChatEndpoint] = _chat_endpoints


###################################
# Embeddings Generation Endpoints #
###################################
_embeddings_endpoints = Register[EmbeddingsEndpoint]({
    "openai_small": OpenAIEmbeddingsEndpoint(model="text-embedding-3-small", dimensions=1536),
    "openai_large": OpenAIEmbeddingsEndpoint(model="text-embedding-3-large", dimensions=3072),
})
REGISTERS[EmbeddingsEndpoint] = _embeddings_endpoints


######################
# Predefined Prompts #
######################
def _build_prompt(yaml_file: str) -> Prompt:
    with open(Path.cwd() / "src" / "ai" / "prompts" / yaml_file, "r") as f:
        data = yaml.safe_load(f)
    prompt = Prompt()
    prompt.deserialize(data)
    return prompt


_prompts = Register[Prompt]({
    "find_embeddings": _build_prompt("find_embeddings.yml"),
    "chat": _build_prompt("chat.yml"),
})
REGISTERS[Prompt] = _prompts

from typing import Set, Dict, Literal, List

from src.ai.chat_endpoint import ChatEndpoint


class PromptPart:
    def __init__(self, text: str, required_variables: Set[str] = None):
        self.text = text
        self.required_variables = required_variables or set()

    def format(self, all_variables: Dict[str, str]) -> str:
        if not all(variable in all_variables for variable in self.required_variables):
            return ""
        return self.text.format(**all_variables)


class Prompt:
    def __init__(self):
        self.error_when_missing: Set[str] = set()
        self.endpoint_str = "gpt-4o-mini",
        self.temperature = 0.7
        self.response_format: Literal["text", "json"] = "text"
        self.parts: List[PromptPart] = []
        self.all_variables: Set[str] = set()

    @property
    def endpoint(self) -> ChatEndpoint:
        from src.registers import REGISTERS  # Avoid circular import
        return REGISTERS[ChatEndpoint][self.endpoint_str]

    def deserialize(self, data: dict):
        self.error_when_missing = set(data["error_when_missing"])
        self.endpoint_str = data["endpoint"]
        self.temperature = data["temperature"]
        self.response_format = data["response_format"]

        self.all_variables = set(self.error_when_missing)
        for prompt in data["prompt_parts"]:
            required_variables = set(prompt.get("required_variables", []))
            self.add_part(prompt["prompt"], required_variables)

    def add_part(self, text: str, required_variables: Set[str] = None):
        self.parts.append(PromptPart(text, required_variables))
        if required_variables:
            self.all_variables.update(required_variables)

    def build(self, variables: Dict[str, str]) -> str:
        missing_variables = self.error_when_missing - variables.keys()
        if missing_variables:
            raise ValueError(f"Missing variables: {missing_variables}")
        return "".join(part.format(variables) for part in self.parts)

    def chat(self, messages: List[Dict[str, str]]):
        return self.endpoint.chat(messages, self.temperature, self.response_format)

from typing import List, Dict

from dataclasses import dataclass

from .prompt_templates import (
    STATIC_SHORT_TEMPLATE,
    STATIC_LONG_TEMPLATE,
    DYNAMIC_SHORT_TEMPLATE,
    DYNAMIC_LONG_TEMPLATE
)

@dataclass
class PersonaConfig:
    category: str
    template: str | None = None

class PersonaRegistry:
    def __init__(self):
        self.personas = {
            "base_persona": PersonaConfig("base"),
            "static_short_persona": PersonaConfig("static", STATIC_SHORT_TEMPLATE),
            "static_long_persona": PersonaConfig("static", STATIC_LONG_TEMPLATE),
            "dynamic_short_persona": PersonaConfig("dynamic", DYNAMIC_SHORT_TEMPLATE),
            "dynamic_long_persona": PersonaConfig("dynamic", DYNAMIC_LONG_TEMPLATE),
        }

    def get_names(self) -> List[str]:
        return list(self.personas.keys())

    def get_static_names(self) -> List[str]:
        return [n for n, p in self.personas.items() if p.category != "dynamic"]

    def get_static_templates(self) -> Dict[str, str]:
        return {n: p.template for n, p in self.personas.items() if p.category == "static" and p.template is not None}

    def get_dynamic_templates(self) -> Dict[str, str]:
        return {n: p.template for n, p in self.personas.items() if p.category == "dynamic" and p.template is not None}

    # def get_static_configs(self) -> List[PersonaConfig]:
    #     return [p for _, p in self.personas.items() if p.category == "static"]

    # def get_dynamic_configs(self) -> List[PersonaConfig]:
    #     return [p for _, p in self.personas.items() if p.category == "dynamic"]

    # def get_configs(self) -> List[PersonaConfig]:
    #     return [p for _, p in self.personas.items()]

    # def get_template(self, persona_type):
    #     return self.personas[persona_type].template

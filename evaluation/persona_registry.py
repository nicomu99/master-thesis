from typing import List, Dict

from dataclasses import dataclass, field

from .prompt_templates import (
    BASE_TEMPLATE,
    STATIC_SHORT_TEMPLATE,
    # STATIC_MEDIUM_TEMPLATE,
    STATIC_LONG_TEMPLATE,
    DYNAMIC_SHORT_TEMPLATE,
    # DYNAMIC_MEDIUM_TEMPLATE,
    DYNAMIC_LONG_TEMPLATE
)


@dataclass
class PersonaConfig:
    """Configuration for defining a persona used in task generation.
    
    Attributes:
        name (str): String identifier of the persona
        category (str): The category of the persona. Can be "base", "static" and "dynamic".
        template (str): The template used to generate personas of this class. Defaults to None.
        answer_column (str): A string identifier of the column in dataframes where the answer of this persona will be
            saved into. Is simply the name of the persona with "persona" changed to "answer".
    """
    name: str
    category: str
    template: str
    answer_column: str = field(init=False)

    def __post_init__(self):
        self.answer_column = self.name.replace("persona", "answer")


class PersonaRegistry:
    """A registry containing persona configurations.
    
    Attributes:
        persona_configs (List[PersonaConfig]): A list containing persona configurations and templates.
    """
    def __init__(self):
        self.persona_configs = [
            PersonaConfig("base_persona", "static", BASE_TEMPLATE),
            PersonaConfig("static_short_persona", "static", STATIC_SHORT_TEMPLATE),
            PersonaConfig("static_long_persona", "static", STATIC_LONG_TEMPLATE),
            PersonaConfig("dynamic_short_persona", "dynamic", DYNAMIC_SHORT_TEMPLATE),
            PersonaConfig("dynamic_long_persona", "dynamic", DYNAMIC_LONG_TEMPLATE),
        ]

    def get_names(self) -> List[str]:
        """Returns all persona names.

        Returns:
            List[str]: List with all persona names.
        """
        return [p.name for p in self.persona_configs]

    def get_static_names(self) -> List[str]:
        """Returns static and base persona names.

        Returns:
            List[str]: A list containing all static persona names.
        """
        return [p.name for p in self.persona_configs if p.category == "static"]

    def get_dynamic_names(self) -> List[str]:
        """Returns dynamic persona names.

        Returns:
            List[str]: List containing all dynamic persona names.
        """
        return [p.name for p in self.persona_configs if p.category == "dynamic"]

    def get_static_templates(self) -> Dict[str, str]:
        """Returns the static persona templates.

        Returns:
            Dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.category == "static"}

    def get_dynamic_templates(self) -> Dict[str, str]:
        """Returns the dynamic persona templates.

        Returns:
            Dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.category == "dynamic"}

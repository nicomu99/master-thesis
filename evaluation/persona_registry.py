from typing import List, Dict

from dataclasses import dataclass

from .prompt_templates import (
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
        category (str): The category of the persona. Can be "base", "static" and "dynamic".
        template (str | None): The template used to generate personas of this class. Defaults to None.
    """
    category: str
    template: str | None = None


class PersonaRegistry:
    """A registry containing persona configurations.
    
    Attributes:
        personas (dict): A dictionary containing persona configurations and templates.
    """
    def __init__(self):
        self.personas = {
            "base_persona": PersonaConfig("base"),
            "static_short_persona": PersonaConfig("static", STATIC_SHORT_TEMPLATE),
            "static_long_persona": PersonaConfig("static", STATIC_LONG_TEMPLATE),
            "dynamic_short_persona": PersonaConfig("dynamic", DYNAMIC_SHORT_TEMPLATE),
            "dynamic_long_persona": PersonaConfig("dynamic", DYNAMIC_LONG_TEMPLATE),
        }

    def get_names(self) -> List[str]:
        """Returns all persona names.

        Returns:
            List[str]: List with all persona names.
        """
        return list(self.personas.keys())

    def get_static_names(self) -> List[str]:
        """Returns static and base persona names.

        Returns:
            List[str]: A list containing all static persona names.
        """
        return [n for n, p in self.personas.items() if p.category != "dynamic"]

    def get_dynamic_names(self) -> List[str]:
        """Returns dynamic persona names.

        Returns:
            List[str]: List containing all dynamic persona names.
        """
        return [n for n, p in self.personas.items() if p.category == "dynamic"]

    def get_static_templates(self) -> Dict[str, str]:
        """Returns the static persona templates.

        Returns:
            Dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {n: p.template for n, p in self.personas.items() if p.category == "static" and p.template is not None}

    def get_dynamic_templates(self) -> Dict[str, str]:
        """Returns the dynamic persona templates.

        Returns:
            Dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {n: p.template for n, p in self.personas.items() if p.category == "dynamic" and p.template is not None}

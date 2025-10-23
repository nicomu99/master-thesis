from typing import List, Dict, Tuple

from dataclasses import dataclass, field

from .utils import PersonaCategory
from .utils import (
    BASE_PERSONA,
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
        category (PersonaCategory): The category of the persona. Can be STATIC or DYNAMIC.
        template (str): The template used to generate personas of this class. Defaults to None.
        answer_column (str): A string identifier of the column in dataframes where the answer of this persona will be
            saved into. Is simply the name of the persona with "persona" changed to "answer".
        judgment_column (str): A string identifier of the dataframe column name, where LLM-as-a-judge judgements of
            the associated persona are stored. Defaults to persona name with "persona" changed to "judgment".
    """
    name: str
    category: PersonaCategory
    template: str
    answer_column: str = field(init=False)
    judgment_column: str = field(init=False)

    def __post_init__(self):
        self.answer_column = self.name.replace("persona", "answer")
        self.judgment_column = self.name.replace("persona", "judgment")


class PersonaRegistry:
    """A registry containing persona configurations.

    Attributes:
        persona_configs (List[PersonaConfig]): A list containing persona configurations and templates.
    """
    def __init__(self):
        self.base_persona_string = BASE_PERSONA

        self.persona_configs = [
            PersonaConfig("base_persona", PersonaCategory.STATIC, BASE_TEMPLATE),
            PersonaConfig("static_short_persona", PersonaCategory.STATIC, STATIC_SHORT_TEMPLATE),
            PersonaConfig("static_long_persona", PersonaCategory.STATIC, STATIC_LONG_TEMPLATE),
            PersonaConfig("dynamic_short_persona", PersonaCategory.DYNAMIC, DYNAMIC_SHORT_TEMPLATE),
            PersonaConfig("dynamic_long_persona", PersonaCategory.DYNAMIC, DYNAMIC_LONG_TEMPLATE),
        ]

    def get_base_persona_string(self) -> str:
        """Returns the base persona string.

        Returns:
            str: Base persona string.
        """
        return self.base_persona_string

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
        return [p.name for p in self.persona_configs if p.category == PersonaCategory.STATIC]

    def get_dynamic_names(self) -> List[str]:
        """Returns dynamic persona names.

        Returns:
            List[str]: List containing all dynamic persona names.
        """
        return [p.name for p in self.persona_configs if p.category == PersonaCategory.DYNAMIC]

    def get_static_templates(self) -> Dict[str, str]:
        """Returns the static persona templates.

        Returns:
            Dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.category == PersonaCategory.STATIC}

    def get_dynamic_templates(self) -> Dict[str, str]:
        """Returns the dynamic persona templates.

        Returns:
            Dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.category == PersonaCategory.DYNAMIC}

    def get_dynamic_configs(self) -> List[PersonaConfig]:
        """Returns the dynamic persona configs.

        Returns:
            List[PersonaConfig]: A list with persona configs.
        """
        return [p for p in self.persona_configs if p.category == PersonaCategory.DYNAMIC]

    def get_configs(self) -> List[PersonaConfig]:
        """Returns all persona configs.

        Returns:
            List[PersonaConfig]: A list with persona configs.
        """
        return self.persona_configs

    def get_config_dict(self) -> Dict[str, PersonaConfig]:
        """Returns a dictionary with persona configs as items and persona identifiers as keys.

        Returns:
            Dict[str, PersonaConfig]: Dictionary with persona configs.
        """
        return {p.name: p for p in self.persona_configs}

    def get_names_and_configs(self) -> Tuple[List[str], List[PersonaConfig]]:
        """Returns a list with persona name idenfifiers and a list with persona configs.

        Returns:
            Tuple[List[str], List[PersonaConfig]]: List with persona names and list with persona configs.
        """
        return self.get_names(), self.persona_configs

from dataclasses import dataclass, field

from .utils import PersonaCategory
from .utils import (
    BASE_PERSONA,
    HELPFUL_PERSONA,
    BASE_TEMPLATE,
    STATIC_SHORT_TEMPLATE,
    STATIC_MEDIUM_TEMPLATE,
    STATIC_LONG_TEMPLATE,
    DYNAMIC_SHORT_TEMPLATE,
    DYNAMIC_MEDIUM_TEMPLATE,
    DYNAMIC_LONG_TEMPLATE,
    BEGINNER_TEACHER_TEMPLATE,
    INTERMEDIATE_TEACHER_TEMPLATE,
    EXPERT_TEACHER_TEMPLATE
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
        judgment_column (str): A string identifier of the dataframe column name, where LLM-as-a-judge judgments of
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

    def is_empty(self) -> bool:
        """Returns true if the category is PersonaCategory.EMPTY.

        Returns:
            bool: True if the category is PersonaCategory.EMPTY, else false.
        """
        return self.category == PersonaCategory.EMPTY

    def is_static(self) -> bool:
        """Returns true if the category is PersonaCategory.STATIC.

        Returns:
            bool: True if the category is PersonaCategory.STATIC, else false.
        """
        return self.category == PersonaCategory.STATIC

    def is_dynamic(self) -> bool:
        """Returns true if the category is PersonaCategory.DYNAMIC.

        Returns:
            bool: True if the category is PersonaCategory.DYNAMIC, else false.
        """
        return self.category == PersonaCategory.DYNAMIC

    def is_static_teacher(self) -> bool:
        """Returns true if the category is PersonaCategory.STATIC_TEACHER.

        Returns:
            bool: True if the category is PersonaCategory.STATIC_TEACHER, else false.
        """
        return self.category == PersonaCategory.STATIC_TEACHER


class PersonaRegistry:
    """A registry containing persona configurations.

    Attributes:
        persona_configs (list[PersonaConfig]): A list containing persona configurations and templates.
    """
    def __init__(self):
        self.base_persona_string = BASE_PERSONA
        self.helpful_persona_string = HELPFUL_PERSONA

        self.persona_configs = [
            PersonaConfig("helpful_persona", PersonaCategory.EMPTY, HELPFUL_PERSONA),
            PersonaConfig("no_persona", PersonaCategory.EMPTY, " "),
            PersonaConfig("base_persona", PersonaCategory.STATIC, BASE_TEMPLATE),
            PersonaConfig("static_short_persona", PersonaCategory.STATIC, STATIC_SHORT_TEMPLATE),
            PersonaConfig("static_medium_persona", PersonaCategory.STATIC, STATIC_MEDIUM_TEMPLATE),
            PersonaConfig("static_long_persona", PersonaCategory.STATIC, STATIC_LONG_TEMPLATE),
            PersonaConfig("dynamic_short_persona", PersonaCategory.DYNAMIC, DYNAMIC_SHORT_TEMPLATE),
            PersonaConfig("dynamic_medium_persona", PersonaCategory.DYNAMIC, DYNAMIC_MEDIUM_TEMPLATE),
            PersonaConfig("dynamic_long_persona", PersonaCategory.DYNAMIC, DYNAMIC_LONG_TEMPLATE),
            PersonaConfig("beginner_teacher_persona", PersonaCategory.DYNAMIC, BEGINNER_TEACHER_TEMPLATE),
            PersonaConfig("intermediate_teacher_persona", PersonaCategory.DYNAMIC, INTERMEDIATE_TEACHER_TEMPLATE),
            PersonaConfig("expert_teacher_persona", PersonaCategory.DYNAMIC, EXPERT_TEACHER_TEMPLATE)
        ]

    def get_reference_config(self) -> PersonaConfig:
        """Returns the reference config for tasks that require judgment.

        For tasks that require a reference, the empty persona was chosen as a baseline.

        Returns:
            PersonaConfig: Reference persona config.
        """
        return self.persona_configs[0]

    def get_base_persona_string(self) -> str:
        """Returns the base persona string.

        Returns:
            str: Base persona string.
        """
        return self.base_persona_string

    def get_names(self) -> list[str]:
        """Returns all persona names.

        Returns:
            list[str]: list with all persona names.
        """
        return [p.name for p in self.persona_configs]

    def get_empty_names(self) -> list[str]:
        """Returns names of empty persona types.

        Returns:
            list[str]: A list containing persona names
        """
        return [p.name for p in self.persona_configs if p.is_empty()]

    def get_static_names(self) -> list[str]:
        """Returns static and base persona names.

        Returns:
            list[str]: A list containing all static persona names.
        """
        return [p.name for p in self.persona_configs if p.is_static()]

    def get_dynamic_names(self) -> list[str]:
        """Returns dynamic persona names.

        Returns:
            list[str]: list containing all dynamic persona names.
        """
        return [p.name for p in self.persona_configs if p.is_dynamic()]

    def get_teacher_static_names(self) -> list[str]:
        """Returns static teacher persona names.

        Returns:
            list[str]: A list containing all static teacher persona configs.
        """
        return [p.name for p in self.persona_configs if p.is_static_teacher()]

    def get_teacher_statics(self) -> list[PersonaConfig]:
        """Returns static teacher persona names.

        Returns:
            list[PersonaConfig]: A list containing all static teacher persona configs.
        """
        return [p for p in self.persona_configs if p.is_static_teacher()]

    def get_empty_templates(self) -> dict[str, str]:
        """Returns the empty persona templates.

        Returns:
            dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.is_empty()}

    def get_static_templates(self) -> dict[str, str]:
        """Returns the static persona templates.

        Returns:
            dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.is_static()}

    def get_dynamic_templates(self) -> dict[str, str]:
        """Returns the dynamic persona templates.

        Returns:
            dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.is_dynamic()}

    def get_teacher_static_templates(self) -> dict[str, str]:
        """Returns the static teacher persona templates.

        Returns:
            dict[str, str]: A dictionary with persona names as items and templates as values.
        """
        return {p.name: p.template for p in self.persona_configs if p.is_static_teacher()}

    def get_dynamic_configs(self) -> list[PersonaConfig]:
        """Returns the dynamic persona configs.

        Returns:
            list[PersonaConfig]: A list with persona configs.
        """
        return [p for p in self.persona_configs if p.is_dynamic()]

    def get_configs(self) -> list[PersonaConfig]:
        """Returns all persona configs.

        Returns:
            list[PersonaConfig]: A list with persona configs.
        """
        return self.persona_configs

    def get_config_dict(self) -> dict[str, PersonaConfig]:
        """Returns a dictionary with persona configs as items and persona identifiers as keys.

        Returns:
            dict[str, PersonaConfig]: Dictionary with persona configs.
        """
        return {p.name: p for p in self.persona_configs}

    def get_answer_columns(self) -> list[str]:
        """Returns all answer columns.

        Returns:
            list[str]: list with answer columns.
        """
        return [p.answer_column for p in self.persona_configs]

    def get_judgment_columns(self) -> list[str]:
        """Returns all judgment columns.

        Returns:
            list[str]: list with judgment columns.
        """
        reference_cfg = self.get_reference_config()
        return [p.judgment_column for p in self.persona_configs if p.name != reference_cfg.name]

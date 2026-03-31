"""Configuration management for CostPlan."""

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for CostPlan."""

    model_config = SettingsConfigDict(
        env_prefix="COSTPLAN_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Token estimation settings
    default_output_ratio: float = Field(
        default=0.6, description="Default ratio for predicting output tokens from input tokens"
    )
    token_estimation_mode: str = Field(
        default="tiktoken", description="Token estimation mode: 'tiktoken' or 'heuristic'"
    )

    # Confidence settings
    confidence_threshold: float = Field(
        default=0.3, description="Threshold for confidence level calculation (30% = Medium)"
    )

    # Storage settings
    database_path: str = Field(default="costplan.db", description="Path to SQLite database file")
    pricing_file_path: str | None = Field(
        default=None, description="Path to pricing JSON file (None = use default)"
    )

    # API settings
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str | None = Field(
        default=None, description="Base URL for OpenAI-compatible API"
    )

    # Anthropic (Claude) API
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY"),
    )

    # Azure OpenAI settings (optional)
    azure_openai_endpoint: str | None = Field(
        default=None, description="Azure OpenAI endpoint URL"
    )
    azure_openai_api_key: str | None = Field(default=None, description="Azure OpenAI API key")

    # Calibration settings
    calibration_window: int = Field(
        default=100, description="Number of recent runs to use for rolling error average"
    )

    def __init__(self, **kwargs):
        """Initialize settings with environment variable support."""
        super().__init__(**kwargs)

        # Expand ~ in database path
        if self.database_path.startswith("~"):
            self.database_path = str(Path(self.database_path).expanduser())

    @classmethod
    def load_from_file(cls, config_file: str) -> "Settings":
        """Load settings from a YAML config file.

        Args:
            config_file: Path to YAML config file

        Returns:
            Settings instance
        """
        import yaml

        with Path(config_file).open(encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def get_database_path(self) -> Path:
        """Get the database path as a Path object.

        Returns:
            Path to database file
        """
        path = Path(self.database_path)
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_api_key(self) -> str | None:
        """Get the API key, checking environment variables.

        Returns:
            API key or None
        """
        return self.openai_api_key or os.getenv("OPENAI_API_KEY")

    def get_base_url(self) -> str | None:
        """Get the base URL for API requests.

        Returns:
            Base URL or None (uses OpenAI default)
        """
        return self.openai_base_url or os.getenv("OPENAI_BASE_URL")

    def get_anthropic_api_key(self) -> str | None:
        """Get Anthropic API key (settings or ANTHROPIC_API_KEY env)."""
        return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

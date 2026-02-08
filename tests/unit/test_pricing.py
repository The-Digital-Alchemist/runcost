"""Unit tests for pricing registry."""

import json
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from costplan.core.pricing import PricingRegistry, PricingNotFoundError


@pytest.fixture
def sample_pricing_file():
    """Create a temporary pricing file for testing."""
    pricing_data = {
        "gpt-4": {
            "input_cost_per_1k_tokens": 0.03,
            "output_cost_per_1k_tokens": 0.06
        },
        "gpt-3.5-turbo": {
            "input_cost_per_1k_tokens": 0.0015,
            "output_cost_per_1k_tokens": 0.002
        }
    }
    
    with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(pricing_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


def test_pricing_registry_initialization(sample_pricing_file):
    """Test pricing registry initialization."""
    registry = PricingRegistry(pricing_file_path=sample_pricing_file)
    assert registry is not None


def test_get_model_pricing(sample_pricing_file):
    """Test getting model pricing."""
    registry = PricingRegistry(pricing_file_path=sample_pricing_file)
    
    input_cost, output_cost = registry.get_model_pricing("gpt-4")
    assert input_cost == 0.03
    assert output_cost == 0.06


def test_get_model_pricing_not_found(sample_pricing_file):
    """Test getting pricing for non-existent model."""
    registry = PricingRegistry(pricing_file_path=sample_pricing_file)
    
    with pytest.raises(PricingNotFoundError):
        registry.get_model_pricing("non-existent-model")


def test_list_supported_models(sample_pricing_file):
    """Test listing supported models."""
    registry = PricingRegistry(pricing_file_path=sample_pricing_file)
    
    models = registry.list_supported_models()
    assert "gpt-4" in models
    assert "gpt-3.5-turbo" in models
    assert len(models) == 2


def test_add_model_pricing(sample_pricing_file):
    """Test adding new model pricing."""
    registry = PricingRegistry(pricing_file_path=sample_pricing_file)
    
    registry.add_model_pricing("test-model", 0.01, 0.02)
    input_cost, output_cost = registry.get_model_pricing("test-model")
    
    assert input_cost == 0.01
    assert output_cost == 0.02


def test_has_model(sample_pricing_file):
    """Test checking if model exists."""
    registry = PricingRegistry(pricing_file_path=sample_pricing_file)
    
    assert registry.has_model("gpt-4") is True
    assert registry.has_model("non-existent") is False


def test_pricing_file_not_found():
    """Test handling of missing pricing file."""
    with pytest.raises(FileNotFoundError):
        PricingRegistry(pricing_file_path="non_existent.json")

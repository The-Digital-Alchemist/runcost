"""Unit tests for cost calculator."""

import pytest
from unittest.mock import Mock

from costplan.core.calculator import CostCalculator, ActualCostResult
from costplan.core.pricing import PricingRegistry


@pytest.fixture
def mock_pricing_registry():
    """Create a mock pricing registry."""
    registry = Mock(spec=PricingRegistry)
    registry.get_model_pricing.return_value = (0.001, 0.002)  # $0.001/1K input, $0.002/1K output
    registry.get_full_pricing.return_value = {
        "input_cost_per_1k_tokens": 0.001,
        "output_cost_per_1k_tokens": 0.002,
    }
    return registry


@pytest.fixture
def calculator(mock_pricing_registry):
    """Create a calculator with mocked pricing registry."""
    return CostCalculator(pricing_registry=mock_pricing_registry)


def test_calculator_initialization():
    """Test calculator initialization."""
    calculator = CostCalculator()
    assert calculator is not None
    assert calculator.pricing_registry is not None


def test_calculate_basic(calculator):
    """Test basic cost calculation."""
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
    }
    
    result = calculator.calculate(usage, "gpt-3.5-turbo")
    
    assert isinstance(result, ActualCostResult)
    assert result.model == "gpt-3.5-turbo"
    assert result.actual_input_tokens == 1000
    assert result.actual_output_tokens == 500
    
    # Cost calculations: (1000/1000) * 0.001 + (500/1000) * 0.002
    assert result.actual_input_cost == 0.001
    assert result.actual_output_cost == 0.001
    assert result.actual_total_cost == 0.002


def test_calculate_missing_fields(calculator):
    """Test calculation with missing usage fields."""
    usage = {"prompt_tokens": 1000}  # Missing completion_tokens
    
    with pytest.raises(ValueError, match="must contain"):
        calculator.calculate(usage, "gpt-4")


def test_calculate_from_tokens(calculator):
    """Test calculation from token counts."""
    result = calculator.calculate_from_tokens(1000, 500, "gpt-4")
    
    assert isinstance(result, ActualCostResult)
    assert result.actual_input_tokens == 1000
    assert result.actual_output_tokens == 500
    assert result.actual_total_cost == 0.002


def test_calculate_error_overestimate(calculator):
    """Test error calculation for overestimate."""
    predicted_cost = 0.005
    actual_cost = 0.004
    
    error = calculator.calculate_error(predicted_cost, actual_cost)
    
    # Error = ((0.005 - 0.004) / 0.004) * 100 = 25%
    assert error == 25.0


def test_calculate_error_underestimate(calculator):
    """Test error calculation for underestimate."""
    predicted_cost = 0.003
    actual_cost = 0.004
    
    error = calculator.calculate_error(predicted_cost, actual_cost)
    
    # Error = ((0.003 - 0.004) / 0.004) * 100 = -25%
    assert error == -25.0


def test_calculate_error_zero_actual(calculator):
    """Test error calculation with zero actual cost."""
    error = calculator.calculate_error(0.005, 0.0)
    assert error == 0.0


def test_calculate_error_perfect(calculator):
    """Test error calculation with perfect prediction."""
    error = calculator.calculate_error(0.004, 0.004)
    assert error == 0.0

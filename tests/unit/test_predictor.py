"""Unit tests for cost predictor."""

import pytest
from unittest.mock import Mock, MagicMock

from costplan.core.predictor import CostPredictor, PredictionResult
from costplan.core.pricing import PricingRegistry
from costplan.core.estimator import TokenEstimator
from costplan.config.settings import Settings


@pytest.fixture
def mock_pricing_registry():
    """Create a mock pricing registry."""
    registry = Mock(spec=PricingRegistry)
    registry.get_model_pricing.return_value = (0.001, 0.002)  # $0.001/1K input, $0.002/1K output
    return registry


@pytest.fixture
def mock_token_estimator():
    """Create a mock token estimator."""
    estimator = Mock(spec=TokenEstimator)
    estimator.estimate_tokens.return_value = 1000  # 1000 tokens
    estimator.estimate_from_messages.return_value = 1000
    return estimator


@pytest.fixture
def predictor(mock_pricing_registry, mock_token_estimator):
    """Create a predictor with mocked dependencies."""
    settings = Settings()
    return CostPredictor(
        pricing_registry=mock_pricing_registry,
        settings=settings,
        token_estimator=mock_token_estimator
    )


def test_predictor_initialization():
    """Test predictor initialization."""
    predictor = CostPredictor()
    assert predictor is not None
    assert predictor.pricing_registry is not None
    assert predictor.settings is not None
    assert predictor.token_estimator is not None


def test_predict_basic(predictor, mock_token_estimator):
    """Test basic prediction."""
    result = predictor.predict("Test prompt", "gpt-3.5-turbo")
    
    assert isinstance(result, PredictionResult)
    assert result.model == "gpt-3.5-turbo"
    assert result.predicted_input_tokens == 1000
    assert result.predicted_output_tokens == 600  # 1000 * 0.6 (default ratio)
    
    # Cost calculations: (1000/1000) * 0.001 + (600/1000) * 0.002
    assert result.predicted_input_cost == 0.001
    assert result.predicted_output_cost == 0.0012
    assert result.predicted_total_cost == 0.0022


def test_predict_with_custom_output_ratio(predictor):
    """Test prediction with custom output ratio."""
    result = predictor.predict("Test prompt", "gpt-4", output_ratio=1.0)
    
    assert result.predicted_output_tokens == 1000  # 1000 * 1.0


def test_predict_confidence_level(predictor):
    """Test confidence level calculation."""
    result = predictor.predict("Test prompt", "gpt-4")
    
    # Default should be Medium with no historical data
    assert result.confidence_level == "Medium"


def test_predict_with_historical_error(predictor):
    """Test prediction with historical error data."""
    # Set low historical error (high confidence)
    predictor.update_historical_error("gpt-4", 5.0)
    
    result = predictor.predict("Test prompt", "gpt-4")
    assert result.confidence_level == "High"


def test_predict_from_messages(predictor, mock_token_estimator):
    """Test prediction from messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    
    result = predictor.predict_from_messages(messages, "gpt-3.5-turbo")
    
    assert isinstance(result, PredictionResult)
    assert result.predicted_input_tokens == 1000
    mock_token_estimator.estimate_from_messages.assert_called_once()


def test_batch_predict(predictor):
    """Test batch prediction."""
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    results = predictor.batch_predict(prompts, "gpt-3.5-turbo")
    
    assert len(results) == 3
    assert all(isinstance(r, PredictionResult) for r in results)


def test_update_historical_error(predictor):
    """Test updating historical error."""
    predictor.update_historical_error("gpt-4", 15.0)
    
    result = predictor.predict("Test", "gpt-4")
    assert result.confidence_percent == 15.0


def test_set_historical_errors(predictor):
    """Test setting multiple historical errors."""
    errors = {
        "gpt-4": 10.0,
        "gpt-3.5-turbo": 20.0,
    }
    predictor.set_historical_errors(errors)
    
    result1 = predictor.predict("Test", "gpt-4")
    result2 = predictor.predict("Test", "gpt-3.5-turbo")
    
    assert result1.confidence_percent == 10.0
    assert result2.confidence_percent == 20.0

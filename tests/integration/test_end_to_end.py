"""Integration tests for end-to-end workflows."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from tempfile import TemporaryDirectory
from pathlib import Path

from costplan.core.predictor import CostPredictor
from costplan.core.calculator import CostCalculator
from costplan.core.executor import ProviderExecutor, ExecutionResult
from costplan.storage.tracker import RunTracker
from costplan.config.settings import Settings


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield str(db_path)


@pytest.fixture
def settings(temp_db):
    """Create test settings."""
    return Settings(database_path=temp_db)


def test_predict_only_workflow(settings):
    """Test prediction-only workflow."""
    # Create predictor
    predictor = CostPredictor(settings=settings)
    
    # Make prediction
    prompt = "Write a short story about AI"
    result = predictor.predict(prompt, "gpt-3.5-turbo")
    
    # Verify prediction
    assert result.predicted_input_tokens > 0
    assert result.predicted_output_tokens > 0
    assert result.predicted_total_cost > 0
    assert result.confidence_level in ["High", "Medium", "Low"]


def test_full_execution_workflow_mocked(settings):
    """Test full prediction -> execution -> calculation -> tracking workflow with mocked API."""
    # Create components
    predictor = CostPredictor(settings=settings)
    calculator = CostCalculator()
    tracker = RunTracker(settings=settings)
    
    # Make prediction
    prompt = "Explain quantum computing"
    prediction = predictor.predict(prompt, "gpt-3.5-turbo")
    
    # Mock execution
    mock_execution = ExecutionResult(
        response_text="Quantum computing is...",
        usage={
            "prompt_tokens": prediction.predicted_input_tokens + 10,
            "completion_tokens": prediction.predicted_output_tokens + 5,
            "total_tokens": prediction.predicted_input_tokens + prediction.predicted_output_tokens + 15,
        },
        raw_response=None,
        model="gpt-3.5-turbo",
        success=True,
    )
    
    # Calculate actual cost
    actual = calculator.calculate(mock_execution.usage, "gpt-3.5-turbo")
    
    # Store run
    run = tracker.store_run(prediction, actual, "gpt-3.5-turbo")
    
    # Verify run was stored
    assert run.id is not None
    assert run.predicted_cost == prediction.predicted_total_cost
    assert run.actual_cost == actual.actual_total_cost
    assert run.error_percent is not None


def test_tracker_statistics(settings):
    """Test tracker statistics calculation."""
    predictor = CostPredictor(settings=settings)
    calculator = CostCalculator()
    tracker = RunTracker(settings=settings)
    
    # Create multiple runs
    for i in range(5):
        prediction = predictor.predict(f"Test prompt {i}", "gpt-3.5-turbo")
        
        # Mock execution with varying accuracy
        mock_execution = ExecutionResult(
            response_text="Response",
            usage={
                "prompt_tokens": prediction.predicted_input_tokens + i * 10,
                "completion_tokens": prediction.predicted_output_tokens + i * 5,
                "total_tokens": prediction.predicted_input_tokens + prediction.predicted_output_tokens + i * 15,
            },
            raw_response=None,
            model="gpt-3.5-turbo",
            success=True,
        )
        
        actual = calculator.calculate(mock_execution.usage, "gpt-3.5-turbo")
        tracker.store_run(prediction, actual, "gpt-3.5-turbo")
    
    # Get statistics
    stats = tracker.get_error_stats(model="gpt-3.5-turbo")
    
    assert stats["sample_count"] == 5
    assert stats["avg_error"] >= 0
    assert "std_dev" in stats


def test_recent_runs_retrieval(settings):
    """Test retrieving recent runs."""
    predictor = CostPredictor(settings=settings)
    tracker = RunTracker(settings=settings)
    
    # Create multiple runs
    for i in range(10):
        prediction = predictor.predict(f"Test {i}", "gpt-3.5-turbo")
        tracker.store_run(prediction, actual=None)
    
    # Get recent runs
    recent = tracker.get_recent_runs(limit=5)
    
    assert len(recent) == 5
    # Should be in reverse chronological order
    assert all(recent[i].timestamp >= recent[i+1].timestamp for i in range(len(recent)-1))


def test_model_filtering(settings):
    """Test filtering runs by model."""
    predictor = CostPredictor(settings=settings)
    tracker = RunTracker(settings=settings)
    
    # Create runs for different models
    for model in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4"]:
        prediction = predictor.predict("Test", model)
        tracker.store_run(prediction, actual=None)
    
    # Get runs for specific model
    gpt4_runs = tracker.get_recent_runs(model="gpt-4")
    
    assert len(gpt4_runs) == 2
    assert all(run.model == "gpt-4" for run in gpt4_runs)


def test_rolling_error_average(settings):
    """Test rolling error average calculation."""
    predictor = CostPredictor(settings=settings)
    calculator = CostCalculator()
    tracker = RunTracker(settings=settings)
    
    # Create runs with known errors
    for i in range(10):
        prediction = predictor.predict("Test", "gpt-3.5-turbo")
        
        # Create actual with 10% error
        actual_tokens = int(prediction.predicted_input_tokens * 1.1)
        mock_execution = ExecutionResult(
            response_text="Response",
            usage={
                "prompt_tokens": actual_tokens,
                "completion_tokens": int(prediction.predicted_output_tokens * 1.1),
                "total_tokens": actual_tokens + int(prediction.predicted_output_tokens * 1.1),
            },
            raw_response=None,
            model="gpt-3.5-turbo",
            success=True,
        )
        
        actual = calculator.calculate(mock_execution.usage, "gpt-3.5-turbo")
        tracker.store_run(prediction, actual, "gpt-3.5-turbo")
    
    # Get rolling average
    rolling_avg = tracker.get_rolling_error_average("gpt-3.5-turbo", window=5)
    
    assert rolling_avg is not None
    assert rolling_avg >= 0


def test_error_propagation(settings):
    """Test error handling propagation through workflow."""
    predictor = CostPredictor(settings=settings)
    
    # Try to predict with non-existent model
    from costplan.core.pricing import PricingNotFoundError
    
    with pytest.raises(PricingNotFoundError):
        predictor.predict("Test", "non-existent-model")


@patch('costplan.core.executor.OpenAI')
def test_executor_error_handling(mock_openai_class, settings):
    """Test executor error handling."""
    # Mock API error
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai_class.return_value = mock_client
    
    executor = ProviderExecutor(api_key="test-key")
    result = executor.execute("Test", "gpt-3.5-turbo")
    
    assert result.success is False
    assert result.error_message is not None

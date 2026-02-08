"""Unit tests for token estimator."""

import pytest
from costplan.core.estimator import TokenEstimator


def test_token_estimator_initialization():
    """Test token estimator initialization."""
    estimator = TokenEstimator(estimation_mode="tiktoken")
    assert estimator.estimation_mode == "tiktoken"


def test_estimate_tokens_with_tiktoken():
    """Test token estimation with tiktoken."""
    estimator = TokenEstimator(estimation_mode="tiktoken")
    
    text = "Hello, world!"
    tokens = estimator.estimate_tokens(text, "gpt-3.5-turbo")
    
    # Should return a positive integer
    assert isinstance(tokens, int)
    assert tokens > 0
    # For this simple text, should be around 3-5 tokens
    assert 2 <= tokens <= 10


def test_estimate_tokens_with_heuristic():
    """Test token estimation with heuristic."""
    estimator = TokenEstimator(estimation_mode="heuristic")
    
    text = "Hello, world!"  # 13 characters
    tokens = estimator.estimate_tokens(text, "any-model")
    
    # Heuristic: len(text) / 4 = 13 / 4 = 3
    assert tokens == 3


def test_estimate_tokens_empty_string():
    """Test token estimation with empty string."""
    estimator = TokenEstimator()
    
    tokens = estimator.estimate_tokens("", "gpt-3.5-turbo")
    assert tokens == 0


def test_estimate_from_messages():
    """Test token estimation from messages."""
    estimator = TokenEstimator(estimation_mode="tiktoken")
    
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    
    tokens = estimator.estimate_from_messages(messages, "gpt-3.5-turbo")
    
    # Should return a positive integer
    assert isinstance(tokens, int)
    assert tokens > 0


def test_estimate_from_messages_empty():
    """Test token estimation from empty messages."""
    estimator = TokenEstimator()
    
    tokens = estimator.estimate_from_messages([], "gpt-3.5-turbo")
    assert tokens == 0


def test_batch_estimate():
    """Test batch token estimation."""
    estimator = TokenEstimator(estimation_mode="heuristic")
    
    texts = ["Hello", "World", "Test"]
    token_counts = estimator.batch_estimate(texts, "any-model")
    
    assert len(token_counts) == 3
    assert all(isinstance(count, int) for count in token_counts)
    assert all(count > 0 for count in token_counts)


def test_tiktoken_fallback_to_heuristic():
    """Test fallback to heuristic when tiktoken fails."""
    estimator = TokenEstimator(estimation_mode="tiktoken")
    
    # Use an unknown model that might cause tiktoken to fail gracefully
    text = "Test text"
    tokens = estimator.estimate_tokens(text, "unknown-model-xyz")
    
    # Should still return a reasonable estimate
    assert isinstance(tokens, int)
    assert tokens > 0


def test_heuristic_accuracy():
    """Test heuristic estimation accuracy."""
    estimator = TokenEstimator(estimation_mode="heuristic")
    
    # 100 characters should give ~25 tokens
    text = "A" * 100
    tokens = estimator.estimate_tokens(text, "any-model")
    
    assert tokens == 25
